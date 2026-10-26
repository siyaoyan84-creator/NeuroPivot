# File-level license scope and exclusions

LICENSE covers only original content authored by Siyao Yan that Siyao Yan has authority to license. The owner has not asserted ownership over every whole file. Other parties retain their copyright; unclear authorship is excluded. No directory is automatically licensed as original.

P identifies implementation traced to an original-project file; N identifies preparation additions; D identifies documentation/metadata. These are provenance categories, not proof of legal title. Only Siyao Yan-authored original portions of P/N/D are covered. T identifies third-party/mixed material excluded from the blanket grant. F identifies facts not claimed as exclusive property. All upstream rights remain unchanged.

| File | Category | Actual source/evidence | Applicable/applicable terms |
|---|---|---|---|
| `.gitattributes` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `.gitignore` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `configs/affectgpt-source-environment.txt` | F | AffectGPT source environment record, third-party dependency facts | Facts not exclusively licensed; expression subject to confirmation |
| `configs/affectgpt.main.yaml` | T | selected AffectGPT/project YAML settings; Config/default/model configuration derived from AffectGPT; not exclusively original | Upstream AffectGPT root Apache-2.0 and applicable nested terms; no blanket NC |
| `configs/folds.json` | N | scripts/run_gate3a_anchored_pivot_v4_grouped_outer5fold.py fixed splits | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `configs/gate4-observed-core.txt` | F | observed original project environment; dependency-version facts | Facts not exclusively licensed; expression subject to confirmation |
| `configs/paths.example.yaml` | N | new portable template for original project stage paths | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `configs/paths.fold5.example.yaml` | N | new portable fold5/anchor01 path template | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `configs/refed_strict_loso_v1.json` | P | original project strict LOSO protocol, retained configuration | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `docs/AFFECTGPT_ASSET_INVENTORY.json` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/AFFECTGPT_SOURCE_AUDIT.json` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/AFFECTGPT_SYMBOL_MAP.json` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/DATA_AND_INTERFACES.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/DELIVERY.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/FINAL_CHECKS.json` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/NEED_FROM_SERVER.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/PIPELINE_CONTRACTS.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/SERVER_CHECK.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/source_manifest.json` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/SOURCE_MAP.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/THIRD_PARTY_DELIVERY.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `docs/VALIDATION.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `examples/affectgpt_integration.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `examples/condition_av.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `examples/encode_eeg.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `licenses/AffectGPT-APACHE-2.0.txt` | T | Preserved component license text; THIRD_PARTY_DELIVERY/source audit | AffectGPT Apache-2.0 |
| `licenses/ImageBind-CC-BY-NC-SA-4.0.txt` | T | Preserved component license text; THIRD_PARTY_DELIVERY/source audit | ImageBind CC-BY-NC-SA-4.0; third-party only |
| `licenses/LAVIS-BSD-3-Clause.txt` | T | Preserved component license text; THIRD_PARTY_DELIVERY/source audit | LAVIS BSD-3-Clause |
| `patches/affectgpt-physio.patch` | T | Verified four-file diff from AffectGPT commit fffca794c6792023234565370e3f69f0488aede9 | Mixed upstream + local delta; per-target details below; excluded from blanket NC |
| `pyproject.toml` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `README.md` | D | Project curation documentation/metadata; cited upstream material retains its own terms | Only Siyao Yan-authored expression covered; third-party/unclear content excluded |
| `requirements-affectgpt.txt` | N | Curation-added CLI/interface/path tooling or dependency declaration; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `requirements.txt` | N | Curation-added CLI/interface/path tooling or dependency declaration; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `scripts/check_affectgpt_real.py` | N | new real check driver reusing production Builder; no model replacement | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/check_core.py` | N | new core check driver, optional original-project comparison | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/init_work.py` | N | new work-directory bootstrap | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/inspect_release_provenance.py` | N | new read-only source-record inspection | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/neuropivot.py` | N | new CLI entry wrapper | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/release_check_utils.py` | N | new release inspection helpers; server feedback adds _report_json_default and dump serialization fix | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/run_pipeline.py` | N | new CLI orchestration around original grouped protocol | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `scripts/server_release_check.sh` | N | new server handoff launcher; server feedback LF normalization, same shell logic | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/cli.py` | N | Curation-added CLI/interface/path tooling or dependency declaration; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/cli_schema.json` | N | Curation-added CLI/interface/path tooling or dependency declaration; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/config.py` | P | src/config.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/data/av_physio_data.py` | P | scripts/train_gate4_affectgpt_pivot_residual_v4.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/build_manifest.py` | P | scripts/build_manifest.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/eav_manifest.py` | P | src/data/eav_manifest.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/mat_header.py` | P | src/data/mat_header.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/normalization.py` | N | source channel_stats and final residual feature_mean/feature_std; extracted normalization contract | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/data/preprocess_eav.py` | P | scripts/build_gate3_eav_eeg_cache.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/preprocess_refed.py` | P | scripts/build_refed_preprocessed_cache.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/refed_loader.py` | P | scripts/run_refed_light_subject.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/refed_manifest.py` | P | src/data/refed_manifest.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/refed_rounding.py` | P | scripts/audit_refed_fnirs_resample_rounding.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/refed_samples.py` | P | scripts/build_refed_sample_registry.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/schema.py` | P | src/data/schema.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/data/trial_metadata.py` | N | project pivot cache alignment fields and grouped split contracts | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/inference/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/inference/evaluator.py` | P | scripts/eval_gate4_grouped_outer01_label_fix_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/inference/export_pivots.py` | P | scripts/build_gate4_eav_grouped_outer01_pivot_cache_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/inference/interfaces.py` | N | project EEG and residual classes; public checkpoint/API wrappers | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/inference/pivot_cache_base.py` | P | scripts/build_gate4_eav_nested_outer03_pivot_cache_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/inference/predict.py` | N | scripts/eval_gate4_grouped_outer01_label_fix_v1.py; public CLI/output wrapper | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/inference/scoring.py` | P | scripts/eval_gate4_grouped_fiveclass_label_fix_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/integrations/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/integrations/affectgpt/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/integrations/affectgpt/affect_plumbing.py` | P | scripts/train_gate4_affectgpt_physio_residual_v2.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/integrations/affectgpt/backend.py` | N | new explicit dependency loader around external Config/registry; imports do not establish copied-code ownership | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/integrations/affectgpt/cache_av.py` | P | scripts/cache_gate4_affectgpt_av_embeddings.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/integrations/affectgpt/inspect.py` | N | new source closure/token/installation inspection | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/models/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/models/eeg.py` | P | src/models/gate3_eeg_pivot_v2.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/models/ordinal.py` | P | src/models/gate3_eeg_pivot_v3.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/models/residual.py` | P | scripts/train_gate4_grouped_outer01_three_conditions_label_fix_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/paths.py` | N | Curation-added CLI/interface/path tooling or dependency declaration; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/preprocessing/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/preprocessing/eav.py` | P | src/preprocessing/eav.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/preprocessing/refed.py` | P | src/preprocessing/refed.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/samples/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/samples/refed.py` | P | src/samples/refed.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/__init__.py` | N | Curation-added package initializer; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `src/neuropivot/training/bridge_base.py` | P | scripts/run_gate3a_anchored_pivot_v4_nested_outer03.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/contracts.py` | N | project grouped anchor/checkpoint provenance; added consistency guards | NC for Siyao Yan original portions only, other/unclear authorship excluded |
| `src/neuropivot/training/protocol.py` | P | scripts/run_gate3a_anchored_pivot_v4_grouped_outer5fold.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/refed_anchor.py` | P | scripts/build_gate3a_refed_anchor_v4.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/refed_source.py` | P | scripts/run_gate3a_refed_target_loso.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/residual.py` | P | scripts/train_gate4_grouped_outer01_three_conditions_label_fix_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `src/neuropivot/training/task_adapter.py` | P | scripts/train_gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1.py (SOURCE_MAP; original file exists) | NC for Siyao Yan original portions only only for owner-confirmed original portions; external contributions not presumed covered |
| `tests/affectgpt_doubles.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `tests/test_affectgpt_contract.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `tests/test_core.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |
| `tests/test_report_serialization.py` | N | Curation-added test/example; recorded public preparation work, not a copied upstream file identified in SOURCE_MAP | Only Siyao Yan-authored additions covered; other/unclear authorship excluded |

| `LICENSE` | D | Current project license/documentation | Own original expression only; exclusions retained |
| `docs/LICENSE_SUMMARY.md` | D | Current project license/documentation | Own original expression only; exclusions retained |
| `docs/LICENSE_SCOPE.md` | D | Current project license/documentation | Own original expression only; exclusions retained |

## Patch targets (not separate newly owned source files)

| Patch target | Evidence | Terms retained / approval boundary |
|---|---|---|
| AffectGPT/config.py | Verified original baseline + PHYSIO constant delta | AffectGPT root Apache-2.0; added lines author/rights not established by diff alone; no new NC restriction on upstream |
| AffectGPT/my_affectgpt/conversation/conversation_video.py | Verified baseline + Chat PHYSIO hunks; imports external processors/ImageBind | AffectGPT root terms plus any incorporated source notices; exact historical nested provenance remains as recorded in THIRD_PARTY_DELIVERY. Imports alone do not apply ImageBind terms to the entire file. No blanket NC |
| AffectGPT/my_affectgpt/models/affectgpt.py | Verified baseline + PHYSIO/projector hunks; inherits Blip2Base | AffectGPT root and applicable incorporated notices; inherited dependencies retain separate terms. No blanket NC |
| AffectGPT/my_affectgpt/models/tokenizer.py | Actual header: Adapted from salesforce@LAVIS, Copyright(c)2023 salesforce.com, SPDX BSD-3-Clause | BSD-3-Clause header retained, plus applicable upstream obligations; project NC does not replace BSD grant. Any local delta needs separate authorship/compatibility confirmation |

Four targets are delivered only by fixed upstream plus patch, not copied wholesale. Local delta copyright ownership cannot be inferred from server Git status or commit ownership. Because delta authorship/compatibility is unresolved, the whole patch remains excluded from the project's NC grant; preservation of that exclusion is not a new permission to redistribute any otherwise uncleared contribution.

External AffectGPT files, LAVIS/MiniGPT, ImageBind, library dependencies, datasets and pretrained weights are not relicensed here; see THIRD_PARTY_DELIVERY.md and licenses/. This file inventories distributed files, not all external dependency files. Unresolved original-project provenance must not be silently described as verified original institutional code.


## Unresolved portions expressly excluded

- P rows: incorporated outside code/contributions not attributable to Siyao Yan; historical source paths alone do not establish title.
- N/D rows: independently owned contributor expression, including the server-supplied serialization fix in scripts/release_check_utils.py, unless Siyao Yan authority over that contribution is established. No entire file is presumed owned.
- patches/affectgpt-physio.patch: all four targets and unclear local-delta rights remain outside the project blanket grant; original BSD/Apache obligations remain. Historical nested NOTICE scope remains as documented.
- configs/affectgpt.main.yaml: upstream-derived settings/expressions retain upstream terms; facts are not claimed exclusively.

These specific exclusions prevent claiming a single fully cleared whole-repository license. They do not cancel the license for identifiable Siyao Yan-authored original content. External data, weights and dependencies remain separate.
