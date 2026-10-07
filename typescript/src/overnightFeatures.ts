/** Explanations for the separate 15-column next-open contract, in model order. */
export const OVERNIGHT_FEATURE_GROUPS: Record<string, string> = {
  prior_close: "Prior sessions", prior_return_1d: "Prior sessions",
  prior_return_5d: "Prior sessions", prior_volatility_5d: "Prior sessions",
  today_open: "Today’s observed open", today_open_gap: "Today’s observed open",
  assumed_close: "Your close scenario", assumed_day_return: "Your close scenario",
  assumed_vs_prior_close: "Your close scenario", weekday: "Calendar",
  day_fit_predicted_return: "Morning model outputs", day_rank_score: "Morning model outputs",
  day_svr_predicted_return: "Morning model outputs", day_svc_direction_margin: "Morning model outputs",
  fit_minus_assumed_day_return: "Morning model outputs",
};

export const OVERNIGHT_FEATURE_DESCRIPTIONS: Record<string, string> = {
  prior_close: "Last completed session’s verified raw close.",
  prior_return_1d: "Return between the two most recent completed closes.",
  prior_return_5d: "Change over five completed close-to-close intervals.",
  prior_volatility_5d: "Standard deviation of five completed close-to-close returns.",
  today_open: "Observed opening price for the scenario session.",
  today_open_gap: "Today’s open divided by the prior close, minus one.",
  assumed_close: "The editable close used only for this scenario.",
  assumed_day_return: "Assumed close divided by today’s open, minus one.",
  assumed_vs_prior_close: "Assumed close divided by the prior close, minus one.",
  weekday: "Exchange session’s weekday number.",
  day_fit_predicted_return: "Open-to-close Fit prediction from the pinned morning estimator.",
  day_rank_score: "Relative Rank score from the pinned morning estimator; not a return.",
  day_svr_predicted_return: "SVR output from the pinned morning estimator.",
  day_svc_direction_margin: "Direction-SVC margin from the pinned morning estimator; not a probability.",
  fit_minus_assumed_day_return: "Fit prediction minus the day return implied by your assumed close.",
};
