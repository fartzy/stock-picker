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
  prior_close: "Previous session’s verified close.",
  prior_return_1d: "Last close-to-close return.",
  prior_return_5d: "Five-session close-to-close return.",
  prior_volatility_5d: "Five-session close-to-close volatility.",
  today_open: "Observed open for this session.",
  today_open_gap: "Today’s open vs. prior close.",
  assumed_close: "Close used for this scenario.",
  assumed_day_return: "Assumed close vs. today’s open.",
  assumed_vs_prior_close: "Assumed close vs. prior close.",
  weekday: "Session weekday.",
  day_fit_predicted_return: "Morning Fit open-to-close prediction.",
  day_rank_score: "Morning Rank score; not a return.",
  day_svr_predicted_return: "Morning SVR output.",
  day_svc_direction_margin: "Morning SVC margin; not a probability.",
  fit_minus_assumed_day_return: "Fit prediction minus assumed day return.",
};
