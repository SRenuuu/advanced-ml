# Energy efficiency: heating load regression findings

Summary of the EDA, model performance, and parameter interpretation from
`assignment_submission.ipynb`.

The dataset covers 768 simulated buildings (ENB2012). We're predicting Heating Load (Y1)
from six quantitative building-geometry features: Relative Compactness (X1), Surface Area
(X2), Wall Area (X3), Roof Area (X4), Overall Height (X5), and Glazing Area (X7). Two coded
variables (Orientation, Glazing Area Distribution) were dropped because their integer labels
aren't quantitative.

## EDA findings

Overall Height (X5) has the strongest positive linear link to Heating Load (r = 0.889), and
Roof Area (X4) has the strongest negative one (r = -0.862). Surface Area (X2) is also
negatively associated (r = -0.658), and Relative Compactness (X1) positively (r = 0.622).
Wall Area (X3) has a moderate positive correlation (r = 0.456), and Glazing Area (X7) a
weaker one (r = 0.270).

Scatter and hexbin plots show clear but not perfectly straight trends, especially for the
geometry variables. A linear model captures the direction of these relationships, but they
aren't perfectly linear.

Skewness is mild across every predictor. Wall Area (0.533) and Relative Compactness (0.496)
are the most skewed, and the IQR rule flags zero outliers on any predictor.

The geometry variables (X1, X2, X4, X5) are heavily entangled with each other. X1 and X2 move
almost in lockstep (r = -0.992), as do X4 and X5 (r = -0.973), and several other pairs sit
above 0.8. This multicollinearity matters most when reading the model's coefficients below.
Glazing Area (X7) is essentially uncorrelated with everything else (r ≈ 0), so it stands apart
from that cluster.

## Model performance

A multiple linear regression was fit on a 75/25 train/test split (random_state=42).

| | R² | RMSE | MAE |
|---|---|---|---|
| Train | 0.915 | 2.92 | 2.02 |
| Test | 0.915 | 2.99 | 2.15 |

The model explains about 91.5% of the variance in Heating Load on both sets, and the test
errors sit close to the training errors, so there's no meaningful over- or under-fitting. A
typical prediction on unseen buildings is off by roughly 2 to 3 kWh/m², small next to Heating
Load's ~10 kWh/m² standard deviation and ~37 kWh/m² range in the data.

## What the coefficients mean, in plain English

Treat the model as a simple rule of thumb: start from a baseline number, then add or subtract
for each feature of the building.

**The intercept, about 86.2, is the model's starting point for a building with every predictor
at zero.** That building doesn't exist, so this number isn't a real prediction on its own; it's
just the anchor the other terms adjust from.

- **Relative Compactness (X1) has a coefficient of about -64.9.** Holding everything else fixed, a
fully more compact building is predicted to need about 65 kWh/m² less heating. A more
cube-like shape loses less heat, so the negative sign makes sense physically. One caveat: X1
is almost a mirror image of Surface Area in this data (r ≈ -0.99), so the model can't cleanly
separate the two, and this number is shaky read on its own.

- **Surface Area (X2) has a coefficient of about -0.065.** Each extra square metre of outer surface
is predicted to lower Heating Load slightly, which looks backwards at first, since more
exposed surface should mean more heat loss. This is the same collinearity problem: surface
area, compactness, roof area, and height all move together in this dataset, so one real effect
gets split across several correlated variables with confusing individual signs. The raw
correlation between Surface Area and Heating Load was clearly negative (r ≈ -0.66) before the
other variables entered the model, so don't read this coefficient as "surface area doesn't
matter."

- **Wall Area (X3) has a coefficient of about +0.038.** Each extra square metre of wall, holding the
rest fixed, adds about 0.04 kWh/m² of heating need. More wall means more surface exposed to
the outside, so a bit more heat escapes. The effect is small, and it's trustworthy because
wall area is fairly independent of the other geometry variables.

- **Roof Area (X4) has a coefficient of about -0.052.** Each extra square metre of roof is predicted
to lower Heating Load, again fighting intuition on its own. Roof area is tightly linked to
height and compactness (r ≈ -0.97 with height), so this coefficient is a mix of effects rather
than the roof's true individual contribution.

- **Overall Height (X5) has a coefficient of about +4.01, the biggest raw-unit effect in the
model.** Each extra metre of height adds about 4 kWh/m² to Heating Load, which matches
intuition: taller buildings have more exposed wall and more volume to heat.

- **Glazing Area (X7) has a coefficient of about +20.21.** Going from the data's smallest glazing
share (0%) to its largest (40%) adds roughly 8 kWh/m². Windows lose heat faster than insulated
walls, so this makes physical sense. Glazing Area isn't correlated with any geometry variable,
so unlike the geometry coefficients above, this one is trustworthy on its own.

Raw coefficients aren't comparable across predictors because they're measured in different
units: a ratio, three separate square-metre areas, metres, and a fraction. Standardising every
predictor and refitting gives a fair ranking of effect size: Overall Height (X5) first, then
Relative Compactness (X1), then Roof Area (X4) and Surface Area (X2), then Glazing Area (X7),
with Wall Area (X3) smallest.

**For an architect, the takeaway is that shape and height dominate Heating Load**. A shorter, more
compact building with modest glazing needs less heating. The individual signs on Surface Area
and Roof Area shouldn't be trusted in isolation; they're too entangled with height and
compactness in this dataset to separate cleanly, so treat shape and height together as the
real driver.
