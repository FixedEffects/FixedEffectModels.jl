# Changelog

## 2.2.0

- Fix regressors and the constant that were dropped as collinear because of the units of another regressor, e.g. the constant with a regressor around 1e12. Fix standard errors in the same cases.
- Fix regressors that were dropped as collinear with fixed effects because of their mean, e.g. `year` in a panel of five years. The test now compares what remains after absorbing the fixed effects to the variation around the mean, up to the accuracy of the solver.
- Fix the degrees of freedom of slopes nested in a cluster: they count, as in reghdfe, so standard errors, the F-statistic, and `F_kp` change in these models.
- Fix the F-statistic when the covariance matrix is singular: it tests the linearly independent coefficients, rather than reporting values as large as 1e18.
- Fix `adjr2` with slope fixed effects and no constant.
- Fix the degrees of freedom absorbed by fixed effects: levels that are redundant across fixed effects no longer count (one per connected component), and fixed effects nested in a cluster count only once, for the constant. With two or more fixed effects, this changes `dof_fes`, `dof_residual`, standard errors, the F-statistic, and adjusted R².
- Fix `F_kp`: it uses the first-stage degrees of freedom when the model is over-identified, applies a degrees-of-freedom correction under simple covariance (it was too high, by more with many fixed effects), and no longer depends on the order of the endogenous variables. `p_kp` changes accordingly.
- Require Vcov 0.8.5, which contains the two changes to the rank test and fixes standard errors of coefficients with small variances, and FixedEffects 3.4.1.
- Improve the results display: show the dependent variable in the header, mark IV models in the title, format the F-test p-value like coefficient p-values, add a thousands separator to the number of observations, and warn on non-convergence instead of showing a Converged row (#284).

## 2.1.0

- Reduce compilation latency: formula parsing, schema and model matrices go through a vector-based term pipeline, so methods compile per term type rather than per formula shape or column names (#283).

## 2.0.0

### Breaking changes

- A standalone RHS slope that is exactly spanned by a continuous-slope fixed effect is now omitted from coefficient output. For example, `fe(id)*x` still includes group intercepts and group-specific slopes, but no longer reports the unidentified common `x` coefficient as a dropped `0` with `NaN` inference statistics. Code that relies on coefficient positions or names should account for the shorter coefficient vector.
