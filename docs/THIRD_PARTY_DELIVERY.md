# Third-party delivery decision

Updated 2026-10-23. Chosen source delivery is **verified fixed upstream plus the minimal local patch**, not the private review bundle or a copied broad source subtree. Installation commands are in README.md. The original checkout is kept untouched and may still be used when explicitly configured. No implicit sibling third_party discovery remains.

## Baseline and delta actually verified

- Official source: https://github.com/zeroQiaoba/AffectGPT ; commit `fffca794c6792023234565370e3f69f0488aede9`, tree `0f4ef8e9ce8c1374b915bdedc8cc1b17231d8291`. The official commit API was successfully read; later API rate limiting did not become evidence of absence.
- Server Git root ends in third_party/AffectGPT; commit/root/status/diff commands all have exit_code=0. The diff is nonempty. Local explicit git-dir/work-tree access now works after removing ambient GIT_* variables. Previous failed inspection is superseded.
- Four text modifications: AffectGPT/config.py, my_affectgpt/conversation/conversation_video.py, my_affectgpt/models/affectgpt.py, my_affectgpt/models/tokenizer.py. They add PHYSIO token/slot, Chat handling and internal physiology branch. Existing fusion resides in the baseline. Bytecode changes, .pre_physio backups, outputs, models and datasets are not included.
- All61 required local-import files exist in the baseline tree. Clean baseline export, git apply --check (exit0), actual application and comparison against modified checkout passed for all61 files. Line endings alone were normalized. No necessary untracked import source was found.
- `patches/affectgpt-physio.patch` and its digest/canonical source fingerprints are recorded in the existing AFFECTGPT_SOURCE_AUDIT.json. No patch was inferred merely from an empty worktree diff.

## Applicable notices and remaining scope limitation

- Root AffectGPT Apache-2.0 text is preserved in licenses/AffectGPT-APACHE-2.0.txt. The newly exported AffectGPT/EmoPrefer/OV-MER license files are Apache texts too; they were not directly copied from private evidence into this tree.
- LAVIS/MiniGPT-derived modules retain BSD headers. Official Salesforce BSD-3-Clause text was obtained from https://raw.githubusercontent.com/salesforce/LAVIS/main/LICENSE.txt and is preserved in licenses/LAVIS-BSD-3-Clause.txt.
- ImageBind-derived modules retain Meta attribution. Its official https://raw.githubusercontent.com/facebookresearch/ImageBind/main/LICENSE is CC-BY-NC-SA4.0, preserved in licenses/ImageBind-CC-BY-NC-SA-4.0.txt. It is **not Apache or unrestricted commercial-use code**. Broad package registration imports this subtree even for the selected HuBERT/CLIP route. Do not remove notices or relicense it under the root Apache text.
- Official license texts were retrieved on2026-10-23; exact URLs and SHA256 are in the source audit. They are current official notices, not proof of the exact historical nested revision. Precise historical LAVIS/MiniGPT/ImageBind provenance and additional applicable NOTICE remain a bounded attribution limitation to confirm with the source owner. It is recorded here and in the consolidated missing-items list, not treated as a new demand for another entire checkout.
- Patch application/source availability is verified separately from complete dependency installation and license-scope interpretation. Follow each component's terms. Foundation weights/data have separate licenses/access terms and are not redistributed.

## Project license boundary

LICENSE applies only to original content authored by Siyao Yan. Third-party, derived and unclear-authorship portions retain their terms or remain excluded as documented in LICENSE_SCOPE.md. The patch is not blanket-relicensed. Existing third-party commercial rights are unchanged. See LICENSE_ZH.md. No custom SPDX identifier is asserted.
