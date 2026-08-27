# Changelog

## 2.0.0

### Breaking changes

- A standalone RHS slope that is exactly spanned by a continuous-slope fixed effect is now omitted from coefficient output. For example, `fe(id)*x` still includes group intercepts and group-specific slopes, but no longer reports the unidentified common `x` coefficient as a dropped `0` with `NaN` inference statistics. Code that relies on coefficient positions or names should account for the shorter coefficient vector.
