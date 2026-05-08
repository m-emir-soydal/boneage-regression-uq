import {
  BarChart,
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const levels = ["0.90", "0.95"];

const baselinePicp = [0.8875, 0.9556];
const multiPicp = [0.8867, 0.9564];
const nominal = [0.9, 0.95];

const baselineMpiw = [334.735, 370.471];
const multiMpiw = [334.674, 370.437];

const baselineMae = 127.266;
const multiMae = 127.267;
const baselineRmse = 133.819;
const multiRmse = 133.818;

const baseline95Gap = baselinePicp[1] - nominal[1];
const multi95Gap = multiPicp[1] - nominal[1];

export default function ConformalUQBaselineVsMulti(): JSX.Element {
  return (
    <Stack gap={20}>
      <H1>Conformal Prediction UQ Report: Baseline vs Multi-input</H1>
      <Text>
        Test split comparison using split-conformal prediction intervals built from validation residuals. This
        canvas explains what each uncertainty metric means and summarizes the current generated outputs:
        {" "}
        <Text as="span" weight="semibold">
          `conformal_metrics_baseline.csv` and `conformal_metrics_multi.csv`
        </Text>
        .
      </Text>

      <Row gap={8} align="center" wrap>
        <Pill tone="info" active>
          n(test)=1262
        </Pill>
        <Pill tone="info" active>
          n(cal)=1261
        </Pill>
        <Pill tone="neutral" active>
          Levels: 0.90, 0.95
        </Pill>
      </Row>

      <Grid columns={4} gap={16}>
        <Stat value={`${(baselinePicp[1] * 100).toFixed(2)}%`} label="Baseline PICP@0.95" tone="info" />
        <Stat value={`${(multiPicp[1] * 100).toFixed(2)}%`} label="Multi PICP@0.95" tone="success" />
        <Stat value={baselineMpiw[1].toFixed(2)} label="Baseline MPIW@0.95 (months)" />
        <Stat value={multiMpiw[1].toFixed(2)} label="Multi MPIW@0.95 (months)" />
      </Grid>

      <Callout tone="info" title="Executive Interpretation">
        Both models are nearly identical under conformal intervals in this run. Coverage at 95% is very close to
        nominal for both, and interval widths are effectively the same. This indicates the conformal calibration step
        dominates interval behavior more than the baseline-vs-multi architecture difference in the current setup.
      </Callout>

      <Divider />

      <H2>What the Metrics Mean</H2>
      <H3>PICP (Prediction Interval Coverage Probability)</H3>
      <Text>
        PICP measures how often the true bone age falls inside the predicted interval. At confidence level alpha, the
        ideal target is PICP ~= alpha. If PICP is below alpha, intervals are under-covering (overconfident). If PICP
        is above alpha, intervals are conservative (wider than necessary).
      </Text>

      <H3>MPIW (Mean Prediction Interval Width)</H3>
      <Text>
        MPIW measures average interval width in months. Smaller MPIW means sharper intervals, but sharpness is only
        useful when PICP is near target. Reliable UQ aims for a good coverage-width tradeoff, not minimum width alone.
      </Text>

      <H3>Conformal Quantile q_hat</H3>
      <Text>
        Split conformal computes residuals on a calibration set and chooses q_hat as an upper quantile. For each test
        prediction y_hat, interval is [y_hat - q_hat, y_hat + q_hat]. This gives finite-sample coverage guarantees
        under exchangeability assumptions.
      </Text>

      <Divider />

      <H2>Headline Metrics</H2>
      <Table
        headers={[
          "Metric",
          "Baseline",
          "Multi-input",
          "Difference (Multi - Baseline)",
          "Interpretation",
        ]}
        rows={[
          [
            "PICP@0.90",
            `${(baselinePicp[0] * 100).toFixed(2)}%`,
            `${(multiPicp[0] * 100).toFixed(2)}%`,
            `${((multiPicp[0] - baselinePicp[0]) * 100).toFixed(2)} pp`,
            "Both close to 90% target.",
          ],
          [
            "PICP@0.95",
            `${(baselinePicp[1] * 100).toFixed(2)}%`,
            `${(multiPicp[1] * 100).toFixed(2)}%`,
            `${((multiPicp[1] - baselinePicp[1]) * 100).toFixed(2)} pp`,
            "Both near nominal 95% coverage.",
          ],
          [
            "MPIW@0.90",
            `${baselineMpiw[0].toFixed(3)} mo`,
            `${multiMpiw[0].toFixed(3)} mo`,
            `${(multiMpiw[0] - baselineMpiw[0]).toFixed(3)} mo`,
            "Sharpness is effectively equal.",
          ],
          [
            "MPIW@0.95",
            `${baselineMpiw[1].toFixed(3)} mo`,
            `${multiMpiw[1].toFixed(3)} mo`,
            `${(multiMpiw[1] - baselineMpiw[1]).toFixed(3)} mo`,
            "Higher confidence increases width as expected.",
          ],
          [
            "MAE",
            `${baselineMae.toFixed(3)} mo`,
            `${multiMae.toFixed(3)} mo`,
            `${(multiMae - baselineMae).toFixed(3)} mo`,
            "Point error unchanged in this run.",
          ],
          [
            "RMSE",
            `${baselineRmse.toFixed(3)} mo`,
            `${multiRmse.toFixed(3)} mo`,
            `${(multiRmse - baselineRmse).toFixed(3)} mo`,
            "Large-error profile unchanged here.",
          ],
        ]}
        striped
      />

      <H2>Coverage vs Nominal Confidence</H2>
      <LineChart
        categories={levels}
        series={[
          { name: "Nominal", data: nominal, tone: "neutral" },
          { name: "Baseline PICP", data: baselinePicp, tone: "info" },
          { name: "Multi PICP", data: multiPicp, tone: "success" },
        ]}
        height={280}
      />
      <Text tone="secondary" size="small">
        Ideal calibration sits on the nominal line. At 95%, baseline gap is {(baseline95Gap * 100).toFixed(2)} pp and
        multi gap is {(multi95Gap * 100).toFixed(2)} pp.
      </Text>

      <H2>Interval Width Tradeoff (MPIW)</H2>
      <BarChart
        categories={levels}
        series={[
          { name: "Baseline MPIW", data: baselineMpiw, tone: "info" },
          { name: "Multi MPIW", data: multiMpiw, tone: "success" },
        ]}
        height={280}
        valueSuffix=" mo"
      />
      <Text tone="secondary" size="small">
        Expected behavior is visible: 95% intervals are wider than 90% intervals. Width differences between baseline
        and multi are negligible in this conformal run.
      </Text>

      <Callout tone="warning" title="How to Use These Metrics for Deployment Decisions">
        Prefer models/settings where PICP is close to target alpha on your target population and MPIW remains
        clinically useful. If coverage is good but intervals are too wide, refine model quality or calibration split;
        if intervals are narrow but under-covering, do not trust uncertainty at face value.
      </Callout>
    </Stack>
  );
}
