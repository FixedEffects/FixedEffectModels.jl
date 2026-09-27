# Changelog

## 2.1.0

- Reduce compilation latency: formula parsing, schema and model matrices go through a vector-based term pipeline, so methods compile per term type rather than per formula shape or column names (#283).

## 2.0.0

### Breaking changes

- A standalone RHS slope that is exactly spanned by a continuous-slope fixed effect is now omitted from coefficient output. For example, `fe(id)*x` still includes group intercepts and group-specific slopes, but no longer reports the unidentified common `x` coefficient as a dropped `0` with `NaN` inference statistics. Code that relies on coefficient positions or names should account for the shorter coefficient vector.
