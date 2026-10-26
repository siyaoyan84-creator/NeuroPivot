# Release delivery

The current publication source is NeuroPivot-release. The historical server handoff archive is kept outside this release; it is not the current source snapshot. The release contains source, configuration, the four-file AffectGPT patch, notices and scripts. It excludes environments, data, weights, caches, logs, raw feedback and private evidence.

Server checking separates frozen inference from training-side gradients. Outer01 uses subject14 trials0/1 for frozen scoring and subject03 trial0 with inner4/job14 cross-fitted physiology for the gradient check. Job14 is a role ID, not EAV subject14. REFED and EAV subject IDs belong to different namespaces.

The real mode initializes one model, scores two cached-AV trials and checks one residual backward without optimizer steps or checkpoint writes. Task-token gradients are optional and were disabled in the completed check. Actual diagnostics and unchanged-asset checks passed; see VALIDATION.md. This is not full training, raw media encoding, a fresh installation qualification or paper-result reproduction.

Set the seven existing asset paths in SERVER_CHECK.md. Use the existing neuro-affect-g4-affectgpt environment without installing or upgrading packages. Outputs go only to unique release_checks directories.

Bridge summaries establish fold1-4 anchor02 and fold5 anchor01 for all30 jobs. They do not identify the final paper cache/checkpoint/evaluation chain. Six historical source gaps remain. Use saved metadata rather than directory names; missing records remain unknown.

LICENSE applies only to Siyao Yan-authored original content. Unclear contributor and patch rights, historical nested NOTICE scope and final paper artifact lineage remain the concrete unresolved items; see LICENSE_SCOPE.md and THIRD_PARTY_DELIVERY.md. The local source was initially committed before publication; refer to Git history for upload status.
