"""
Implementation of self attention layers:
https://github.com/zhouhaoyi/Informer2020/blob/main/models/attn.py

Informer2020 License from https://github.com/zhouhaoyi/Informer2020/blob/main/LICENSE, accessed
on  May 25, 2022:
'                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'
"""

import math

import numpy as np
import torch
import torch.nn as nn

from darts.models.components.mem_attention import MemAttention, default, reshape_dim


class FullAttention(nn.Module):
    def __init__(
            self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, **kwargs
    ):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attention_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attention_mask is None:
                attention_mask = triangular_causal_mask(B, L, device=queries.device)
            scores.masked_fill_(attention_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class ProbSparseAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, queries, keys, sample_k, n_top):
        B, H, L_K, E = keys.shape
        _, _, L_Q, _ = queries.shape

        # calculate the sampled Q_K
        K_expand = keys.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor * ln(L_K)) * L_Q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = (queries.unsqueeze(-2) @ K_sample.transpose(-2, -1)).squeeze()

        # find the top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = queries[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor * ln(L_Q)
        Q_K = Q_reduce @ keys.transpose(-2, -1)  # factor * ln(L_Q) * L_K

        return Q_K, M_top

    def _get_initial_context(self, values, L_Q):
        B, H, L_V, D = values.shape
        if not self.mask_flag:
            V_mean = values.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, V_mean.size(-1)).clone()
        else:
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            context = values.cumsum(dim=-2)
        return context

    def _update_context(self, context, values, scores, index, L_Q, attention_mask):
        B, H, L_V, D = values.shape

        if self.mask_flag:
            attention_mask = prob_mask(B, H, L_Q, index, scores, device=values.device)
            scores.masked_fill_(attention_mask, -np.inf)

        attention = torch.softmax(scores, dim=-1)

        context[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = (
            attention @ values
        ).type_as(context)
        if self.output_attention:
            attentions = (torch.ones(B, H, L_V, L_V) / L_V).type_as(attention)
            attentions[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attention
            return context, attentions
        return context, None

    def forward(self, queries, keys, values, attention_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = int(self.factor * np.ceil(np.log(L_K)))  # c * ln(L_K)
        u = int(self.factor * np.ceil(np.log(L_Q)))  # c * ln(L_Q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attention = self._update_context(
            context, values, scores_top, index, L_Q, attention_mask
        )

        return context.transpose(2, 1).contiguous(), attention


class LogSparseAttention(nn.Module):
    """
    Log Sparse Attention
    source: [1]

    [1] Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y. X., & Yan, X. (2019).
    Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting.
    arXiv preprint arXiv:1907.00235.
    """

    def __init__(
            self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, sub_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, win_len, sub_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        log_l = math.ceil(np.log2(sub_len))

        mask = torch.zeros((win_len), dtype=torch.float)
        if (win_len // sub_len) * 2 * (log_l) > index:
            mask[: (index + 1)] = 1
        else:
            while index >= 0:
                if (index - log_l + 1) < 0:
                    mask[:index] = 1
                    break
                mask[index - log_l + 1: (index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2 ** i
                    if (index - new_index) <= sub_len and new_index >= 0:
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def forward(self, queries, keys, values, attention_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
        # if attention_mask is None:
        #     attention_mask = TriangularCausalMask(B, L, device=queries.device)
        # scores.masked_fill_(attention_mask.mask, -np.inf)
        mask = self.log_mask(L, S)
        mask_tri = mask[:, :, : scores.size(-2), : scores.size(-1)]
        scores = scores.to(queries.device)
        mask_tri = mask_tri.to(queries.device)
        scores = scores * mask_tri + -1e9 * (1 - mask_tri)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.dim_heads = d_model // n_heads
        self.mix = mix

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, queries, keys, values, attention_mask, memories=None, pos_emb=None):
        H = self.n_heads
        if isinstance(self.inner_attention, MemAttention) is False:
            B, L, _ = queries.shape
            _, S, _ = keys.shape

            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values).view(B, S, H, -1)

            out, attention = self.inner_attention(queries, keys, values, attention_mask)
            if self.mix:
                out = out.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)

            return self.out_projection(out), attention
        else:
            b, t, e = queries.shape

            memories = default(memories, (None, None))
            mem, lmem = memories

            init_mem = lambda: torch.empty(b, 0, e).to(queries.device)
            mem = default(mem, init_mem)
            lmem = default(lmem, init_mem)
            mem_len, lmem_len = map(lambda t: t.shape[1], (mem, lmem))

            kv_input = torch.cat((lmem, mem, queries), dim=1)

            queries = self.query_projection(queries)

            kv_len = kv_input.shape[1]
            keys = self.key_projection(kv_input)
            values = self.value_projection(kv_input)

            merge_heads = lambda x: reshape_dim(x, -1, (-1, self.dim_heads)).transpose(1, 2)
            q, k, v = map(merge_heads, (queries, keys, values))
            k, v = map(lambda x: x.expand(-1, H, -1, -1), (k, v))

            out = self.inner_attention(
                q,
                k,
                v,
                attention_mask,
                pos_emb,
                kv_len,
                mem_len,
                lmem_len
            )
            return self.out_projection(out), None


