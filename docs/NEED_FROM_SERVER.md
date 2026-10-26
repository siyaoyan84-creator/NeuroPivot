> Current status (2026-10-24): server minimal real initialization, cached-AV scoring and training-side residual backward passed; task-token gradient was disabled. Earlier unexecuted statements below describe the pre-handoff state. See VALIDATION.md for the current verified scope; six historical provenance gaps remain.

# Remaining release evidence

Updated 2026-10-23. Private release_evidence files were read in place and not copied here. All 30 grouped summaries and successful Git command statuses are now available. The modified AffectGPT import closure is present, and its four-source-file delta is verified against the official commit and a clean baseline. No entire runtime, model cache or review_metadata directory is requested.

## Bridge summaries covered; final result lineage unconfirmed

Folds1–4 jobs10–15/20–25/30–35/40–45 use REFED anchor02; fold5 jobs50–55 use anchor01. All are explicit parsed summary values, seed42/ordinal_correct; splits match the public generator. These data supersede the previous request for all 30 summaries. Actual source checkpoint and anchor file paths are absent from every summary. Code-derived paths are not substituted as observed paths.

For exact weight/path provenance only, retrieve a final launch/config/command record or a small extracted checkpoint-metadata JSON for one representative role per fold (then other roles only if they differ). Inspect these precise original locations:

- `checkpoints/gate3/gate3a_anchored_pivot_v4_grouped_outer5fold_v1/eav_test{job_id:02d}_ordinal_correct_seed042_tmm_grouped_v1/` for the executed command/config or metadata beside best.pt; job IDs are10–15,20–25,30–35,40–45,50–55.
- The source/anchor patterns derived from the retained runner are `checkpoints/gate3/gate3a_refed_target_loso_v2/refed_test{AA:02d}_separate_seed042_pilot/best.pt` and `features/gate3/gate3a_refed_anchor_v4/refed_test{AA:02d}_seed042.npz`; AA=2 for folds1–4 and AA=1 for fold5. These are **derived expectations**, not paths stored in the supplied summaries. Transfer just saved arguments/identity fields or a small launch record, not weight tensors.

If the paper claims a single anchor across five folds, the current evidence contradicts that claim: the owner must clarify which runs/results belong to the final table, or provide the corrected final run records. Do not silently relabel fold5 as anchor02 or silently retrain it.

## Dataset identity and exclusion explanation

REFED and EAV subject IDs are different namespaces; every summary says cross_dataset_subject_pairing=false. EAV01 is excluded,02 is validation,03 remains training-side only,04–42 form outer evaluation. These rules agree with code and all summaries. A manuscript/protocol statement explaining these exclusions is needed if not already documented; no biological identity equivalence is inferred from matching numbers.

## Third-party notice/provenance scope

The supplied LICENSE files for AffectGPT/EmoPrefer/OV-MER are Apache texts; none resolves the LAVIS/MiniGPT BSD or ImageBind notice by itself. Official LAVIS and ImageBind license sources are identified in THIRD_PARTY_DELIVERY.md. The exact historical nested revisions/applicability and any additional NOTICE/attribution supplied with those derived files should still be confirmed. Minimum requested material is a small provenance/NOTICE record tied to `third_party/AffectGPT/AffectGPT/my_affectgpt/models/ImageBind/` and LAVIS/MiniGPT-derived modules, not another broad source export. If no such record exists on the server, report that limitation rather than manufacture one.

The previous broad Git-record request is closed: root, commit, status and nonempty diff all succeeded. No necessary untracked source appears in the inspected61-file import closure. Public patch delivery now has a verified baseline; official-source network fetching remains dependent on the user's Git/network setup.

## Execution prerequisites, not server-download requests

A qualified complete AffectGPT/CUDA environment, real REFED/EAV and valid base/stage weights are still required for end-to-end execution. Existing local weight file headers were checked, but full model loading was not. CPU core software was installed and real tests passed; do not request duplicate model downloads or all old runtime outputs. The custom noncommercial LICENSE applies only to Siyao Yan original content; see LICENSE_SCOPE.md for unresolved portions.


For the next server check, follow SERVER_CHECK.md rather than exporting another broad evidence package. Minimum still-needed human confirmation is the identity of each paper-final evaluation asset and its actual checkpoint/cache/run-ID links. A passing real computation check does not establish these historical links. Use the read-only provenance tool first; if fields were never saved, retain the conclusion “cannot determine from existing records”. No runtime-tree transfer is requested.
