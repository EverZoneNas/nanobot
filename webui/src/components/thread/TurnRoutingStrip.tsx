import { useTranslation } from "react-i18next";

import { cn } from "@/lib/utils";
import type { TurnRoutingInfo } from "@/lib/types";

function routingDecisionLabel(
  info: TurnRoutingInfo,
  t: ReturnType<typeof useTranslation>["t"],
): string | null {
  switch (info.decisionReason) {
    case "kept_for_cache":
    case "no_candidate_affinity":
    case "classifier_fallback":
      return t("thread.composer.routingCacheRetained", {
        defaultValue: "Warm prompt cache retained",
      });
    case "same_cache_identity":
      return t("thread.composer.routingSameModel", {
        defaultValue: "Preset adjusted without changing the cached model",
      });
    case "switched_score":
      return t("thread.composer.routingSwitched", {
        defaultValue: "Switched models for this task",
      });
    case "initial_candidate":
      return t("thread.composer.routingInitialCandidate", {
        defaultValue: "Initial route from classifier",
      });
    case "initial_baseline":
      return t("thread.composer.routingInitialBaseline", {
        defaultValue: "Using baseline model",
      });
    case "candidate_unchanged":
      return t("thread.composer.routingCandidateUnchanged", {
        defaultValue: "Route unchanged",
      });
    case "no_candidate_baseline":
      return t("thread.composer.routingNoCandidateBaseline", {
        defaultValue: "No matching rule; kept baseline",
      });
    case "deterministic_rule":
      return t("thread.composer.routingDeterministicRule", {
        defaultValue: "Matched routing rule",
      });
    case "dream_override":
      return t("thread.composer.routingDreamOverride", {
        defaultValue: "Dream model override",
      });
    default:
      return info.decisionReason ?? null;
  }
}

export function TurnRoutingStrip({
  info,
  variant = "message",
}: {
  info?: TurnRoutingInfo | null;
  variant?: "composer" | "message";
}) {
  const { t } = useTranslation();
  if (!info) return null;

  const routeParts = [
    info.modelPreset
      ? t("thread.composer.routingPreset", {
          preset: info.modelPreset,
          defaultValue: "preset {{preset}}",
        })
      : null,
    t("thread.composer.routingModel", {
      model: info.modelName,
      defaultValue: "model {{model}}",
    }),
  ].filter(Boolean);
  const classifierParts = [
    info.taskKind
      ? t("thread.composer.routingKind", {
          kind: info.taskKind,
          defaultValue: "kind {{kind}}",
        })
      : null,
    info.taskType
      ? t("thread.composer.routingType", {
          type: info.taskType,
          defaultValue: "type {{type}}",
        })
      : null,
    info.complexity
      ? t("thread.composer.routingComplexity", {
          complexity: info.complexity,
          defaultValue: "complexity {{complexity}}",
        })
      : null,
  ].filter(Boolean);
  const decisionLabel = routingDecisionLabel(info, t);
  const candidateParts = [
    info.candidateModelPreset
      ? t("thread.composer.routingCandidatePreset", {
          preset: info.candidateModelPreset,
          defaultValue: "candidate preset {{preset}}",
        })
      : null,
    info.candidateModelName
      ? t("thread.composer.routingCandidateModel", {
          model: info.candidateModelName,
          defaultValue: "candidate model {{model}}",
        })
      : null,
  ].filter(Boolean);
  const scoreParts = [
    info.switchScore != null
      ? t("thread.composer.routingSwitchScore", {
          score: info.switchScore.toFixed(2),
          defaultValue: "score {{score}}",
        })
      : null,
    info.qualityBenefit != null
      ? t("thread.composer.routingQualityBenefit", {
          value: info.qualityBenefit.toFixed(2),
          defaultValue: "quality {{value}}",
        })
      : null,
    info.cachePenalty != null
      ? t("thread.composer.routingCachePenalty", {
          value: info.cachePenalty.toFixed(2),
          defaultValue: "cache penalty {{value}}",
        })
      : null,
    info.estimatedReusableTokens
      ? t("thread.composer.routingReusableTokens", {
          count: info.estimatedReusableTokens,
          defaultValue: "{{count}} reusable tokens",
        })
      : null,
  ].filter(Boolean);

  return (
    <div
      className={cn(
        "overflow-hidden rounded-[16px] border px-3 py-2",
        "border-sky-500/18 bg-sky-500/[0.05] text-[11.5px] text-sky-950/85",
        "dark:border-sky-400/20 dark:bg-sky-400/[0.08] dark:text-sky-100/90",
        variant === "composer" ? "composer-status-strip mx-3 mt-3" : "mt-2",
      )}
      role="status"
      aria-label={t("thread.composer.routingStripAria", {
        defaultValue: "Model routing details",
      })}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 leading-5">
        <span className="font-semibold uppercase tracking-[0.08em] text-sky-700/80 dark:text-sky-200/80">
          {t("thread.composer.routingStripTitle", { defaultValue: "Model route" })}
        </span>
        <span>{routeParts.join(" · ")}</span>
        {classifierParts.length > 0 ? (
          <>
            <span className="text-sky-800/35 dark:text-sky-200/35" aria-hidden>·</span>
            <span>{classifierParts.join(" · ")}</span>
          </>
        ) : null}
      </div>
      {decisionLabel || candidateParts.length > 0 || scoreParts.length > 0 ? (
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 leading-5 text-sky-900/70 dark:text-sky-100/75">
          {decisionLabel ? <span>{decisionLabel}</span> : null}
          {candidateParts.length > 0 ? (
            <>
              {decisionLabel ? (
                <span className="text-sky-800/35 dark:text-sky-200/35" aria-hidden>·</span>
              ) : null}
              <span>{candidateParts.join(" · ")}</span>
            </>
          ) : null}
          {scoreParts.length > 0 ? (
            <>
              {decisionLabel || candidateParts.length > 0 ? (
                <span className="text-sky-800/35 dark:text-sky-200/35" aria-hidden>·</span>
              ) : null}
              <span>{scoreParts.join(" · ")}</span>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
