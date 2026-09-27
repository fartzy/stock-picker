/** Trading hero button while a morning score is in flight (click or 8:30 job). */

export const CHECK_PRICES_LABEL = "Check this morning's prices";
export const SCORING_LABEL = "Scoring this morning's prices…";

export function morningScanButton(running: boolean): { label: string; disabled: boolean } {
  return running
    ? { label: SCORING_LABEL, disabled: true }
    : { label: CHECK_PRICES_LABEL, disabled: false };
}
