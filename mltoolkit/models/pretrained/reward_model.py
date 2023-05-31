"""
this file contains code for the model used to compute rewards for the rl_extractive trainer
"""
# external imports
from sentence_transformers import SentenceTransformer
import torch

class RewardModel:
    
    def __init__(self, model_name='all-mpnet-base-v1'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, sent_mat, a, b, a_minus_b, a_int_b, b_minus_a):
        pass

    def _compute_reward(
        self, sent_mat, a_sent_mat, b_sent_mat, a_minus_b, a_int_b, b_minus_a
    ) -> torch.Tensor:
        """Computes the reward scores"""
        # set last visible gpu as device for reward calculation
        last_gpu = (
            len(eutils.get_available_gpus()) - 1
        )  # how many gpus are available -1 is the index of the last gpu
        device = f"cuda:{last_gpu}" if torch.cuda.is_available() else "cpu"
        # if self.reward_gpu is None:
        #     last_gpu = (
        #         len(eutils.get_available_gpus()) - 1
        #     )  # how many gpus are available -1 is the index of the last gpu
        #     device = f"cuda:{last_gpu}" if torch.cuda.is_available() else "cpu"
        # else:
        #     device = f"cuda:{self.reward_gpu}"

        batch_sz = sent_mat.shape[0]

        # #####################################################
        # Document level reward
        def _doc_level_reward(
            partition1: np.ndarray, partition2: np.ndarray
        ) -> torch.Tensor:
            def convert_to_document(sel_sent_mat: np.ndarray) -> list:
                return [" ".join((" ".join(ii)).strip().split()) for ii in sel_sent_mat]

            # Convert sentence matrix to list
            part1_docs, part2_docs = convert_to_document(
                partition1
            ), convert_to_document(partition2)
            assert len(part1_docs) == len(part2_docs), "Batch length mismatch. Check"

            # encode sentence matrix
            flatten_docs = part1_docs + part2_docs
            flatten_embeds = self.reward_model.encode(
                flatten_docs,
                batch_size=self.hparams.reward_batch_size,
                device=device,
                convert_to_tensor=True,
            )
            part1_embeds, part2_embeds = torch.split(flatten_embeds, batch_sz, dim=0)
            assert (
                part1_embeds.shape == part2_embeds.shape
            ), "Size mismatch in doc reward computation"

            # create mask to handle empty docs
            mask = torch.tensor(
                list(map(lambda x: 1 if x else 0, flatten_docs)), device=device
            )
            mask1, mask2 = torch.split(mask, batch_sz, dim=0)
            mask = (
                mask1 * mask2
            )  # if any doc is empty string it would make the similarity 0

            # cosine similarity
            return (
                nn.functional.cosine_similarity(part1_embeds, part2_embeds, dim=1)
                * mask
            ).cpu()

        # #####################################################
        # Sent level reward
        def _sent_level_reward(partition1: np.ndarray, partition2: np.ndarray):
            def _compute_mean(sim_tensor: torch.Tensor) -> float:
                # mean for empty tensor returns nan. Set it to 0
                return (
                    0.0 if sim_tensor.nelement() == 0 else torch.mean(sim_tensor).item()
                )

            # Returns non-empty sentences count across batch
            def _get_sent_count_in_each_partition(partitions: tuple) -> list:
                out = []
                for part in partitions:
                    non_zeros = part.nonzero()[
                        0
                    ]  # since we only want to count non-empty str across cols
                    # in case some rows have all zeros, that won't show up. So explicitly start from 0
                    tmp = np.zeros(len(part), dtype=np.int)
                    eles, cts = np.unique(non_zeros, return_counts=True)
                    tmp[eles] = cts
                    out.append(tmp)
                return out

            part1_count, part2_count = _get_sent_count_in_each_partition(
                (partition1, partition2)
            )
            # flatten partitions (raterization, first across cols (for a sample), then across rows (batch))
            flatten_sents = (
                partition1[partition1.nonzero()].tolist()
                + partition2[partition2.nonzero()].tolist()
            )
            # get embeddings
            if len(flatten_sents) > 0:  # atleast one sentence in two partitions
                flatten_embeds = self.reward_model.encode(
                    flatten_sents,
                    batch_size=self.hparams.reward_batch_size,
                    device=device,
                    convert_to_tensor=True,
                )
                # split returns tuple of tensors
                part1_embeds, part2_embeds = torch.split(
                    flatten_embeds, [sum(part1_count), sum(part2_count)], dim=0
                )

                # split for sentence in each sample doc in batch (derastrize in a way)
                part1_sent_embeds: tuple = torch.split(
                    part1_embeds, part1_count.tolist(), dim=0
                )
                part2_sent_embeds: tuple = torch.split(
                    part2_embeds, part2_count.tolist(), dim=0
                )

                # Note: no need to create mask here since that is taken care by the split function itself.
                #       For empty doc, it returns tensor fo shape [0, 768]

                # cosine similarity  (indexing over batch dim here)
                # by default torch tensor is created on the cpu
                return torch.tensor(
                    [
                        _compute_mean(util.cos_sim(ii, jj))
                        for ii, jj in zip(part1_sent_embeds, part2_sent_embeds)
                    ]
                )
            else:  # literally no sentence in both the partitions
                return torch.zeros(batch_sz)  # return 0 similarity vec of length batch

        doc_rewards = {}
        sent_rewards = {}
        # #####################################################
        # Reward equation I (Negative reward Terms)
        partition_dict = {
            "a_minus_b": a_minus_b,
            "a_int_b": a_int_b,
            "b_minus_a": b_minus_a,
        }

        for comb in itertools.combinations(partition_dict.keys(), 2):
            new_key = "__".join(comb) + "__neg"
            doc_rewards[f"doc__{new_key}"] = _doc_level_reward(
                *tuple(map(lambda key: partition_dict[key], comb))
            )

            sent_rewards[f"sent__{new_key}"] = _sent_level_reward(
                *tuple(map(lambda key: partition_dict[key], comb))
            )

        # Reward equation II (Positive reward Terms)
        doc_rewards["doc__a__a_int_b__pos"] = _doc_level_reward(a_sent_mat, a_int_b)
        doc_rewards["doc__b__a_int_b__pos"] = _doc_level_reward(b_sent_mat, a_int_b)

        sent_rewards["sent__a__a_int_b__pos"] = _sent_level_reward(a_sent_mat, a_int_b)
        sent_rewards["sent__b__a_int_b__pos"] = _sent_level_reward(b_sent_mat, a_int_b)

        # Reward equation III (Negative reward Terms)
        doc_rewards["doc__a__b_minus_a__neg"] = _doc_level_reward(a_sent_mat, b_minus_a)
        doc_rewards["doc__b__a_minus_b__neg"] = _doc_level_reward(b_sent_mat, a_minus_b)

        sent_rewards["sent__a__b_minus_a__neg"] = _sent_level_reward(
            a_sent_mat, b_minus_a
        )
        sent_rewards["sent__b__a_minus_b__neg"] = _sent_level_reward(
            b_sent_mat, a_minus_b
        )

        # Best Weight Combination based on Pearson Correlation Results
        # Custom weights
        # Note: I have added a big constant to make everything positive. Range is [2, -2]
        return 0.25 * (  # doc
            0.75  # doc_pos
            * (
                0.25 * doc_rewards["doc__a__a_int_b__pos"]
                + 0.75 * doc_rewards["doc__b__a_int_b__pos"]
            )
            + 0.25  # doc_neg
            * (-1)
            * (
                0.1 * doc_rewards["doc__a_minus_b__a_int_b__neg"]
                + 0.0 * doc_rewards["doc__a_minus_b__b_minus_a__neg"]
                + 0.9 * doc_rewards["doc__a_int_b__b_minus_a__neg"]
                + 0.0 * doc_rewards["doc__a__b_minus_a__neg"]
                + 0.0 * doc_rewards["doc__b__a_minus_b__neg"]
            )
        ) + 0.75 * (  # sent
            0.7  # sent_pos  # TODO: try high value of this
            * (
                0.5 * sent_rewards["sent__a__a_int_b__pos"]
                + 0.5 * sent_rewards["sent__b__a_int_b__pos"]
            )
            + 0.3  # sent_neg  # 0.5
            * (-1)
            * (
                0.0 * sent_rewards["sent__a_minus_b__a_int_b__neg"]
                + 0.1 * sent_rewards["sent__a_minus_b__b_minus_a__neg"]
                + 0.0 * sent_rewards["sent__a_int_b__b_minus_a__neg"]
                + 0.8 * sent_rewards["sent__a__b_minus_a__neg"]
                + 0.1 * sent_rewards["sent__b__a_minus_b__neg"]
            )
        )
