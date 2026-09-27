# Environment and pinning policy

Section 7 of the [registered protocol](DESIGN.md): pin the actual package versions,
source commit, calendar version and hardware metadata **before any results are
examined**. Nothing here is a version this project guessed; every figure is read from
installed metadata by `chronos2_signal.provenance.capture_environment`.

## Current state

`requirements.txt` pins the core research environment to the versions the invariant
suite was actually executed against:

```
numpy==2.4.6
pandas==3.0.6
scikit-learn==1.9.1
pandas-market-calendars==5.4.0
pyarrow==25.0.1
PyYAML==6.0.1
pytest==9.1.1
```

Python 3.11.15, Linux x86-64, CPU only. Reproduce with:

```
chronos2-signal verify --json
```

The **provider** extra (`yfinance`) and the **model** extra
(`chronos-forecasting`, `torch`) are deliberately *not* pinned yet. They are locked at
build-order step 2, before any result is examined, because their versions are part of
the provenance of every snapshot and every forecast they produce. Until then they are
absent, and the pipeline refuses to improvise:

- a live fetch without `yfinance` raises with the extra to install;
- a Chronos variant without `chronos-forecasting` raises rather than fabricating a
  forecast.

## Version-sensitive behaviour the code handles explicitly

Two places would otherwise silently depend on someone else's default.

**scikit-learn's logistic penalty.** Version 1.8 deprecated the `penalty` keyword in
favour of `l1_ratio`. `decision._l2_logistic` inspects the constructor signature and
passes `l1_ratio=0.0` where supported and `penalty="l2"` otherwise, so the registered
L2 penalty stays explicit on both. Relying on the library default would turn a
registered setting into an assumption.

**The Chronos-2 call signature.** `forecaster.Chronos2Forecaster._introspect` reads the
real `predict_quantiles` signature at load time and records it in the run manifest,
rather than assuming the keyword names. If the installed package does not match, it
raises with the signature it actually found — which is exactly the compatibility
problem the protocol requires resolving before any run. `chronos2-signal smoke`
surfaces that record.

## What a release manifest binds together

`provenance.ReleaseManifest` records, for every reported result: the design version and
its fingerprint, the code revision, the checkpoint id, the pinned 40-character weight
revision and its recorded SHA256, the exchange-calendar package version, the full
environment fingerprint, the three cost scenarios, and the frozen universe hash. A
result without a manifest is not reportable, because there is no way to say afterwards
what produced it.

## Checkpoint verification

The pinned revision is `95a9710e2596287d08352589f42634fa5abdf0a7` and the recorded
safetensors SHA256 is
`ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`. A tag is not
acceptable in place of the commit: a tag can move, and a stored forecast would then no
longer describe its inputs. When the weight file can be located locally the adapter
verifies the hash and raises on a mismatch instead of silently switching weights. When
it cannot be located, that fact is recorded as `weights_verified: not_located` rather
than reported as a pass.
