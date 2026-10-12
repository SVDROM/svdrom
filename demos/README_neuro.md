# EEG compression and reconstruction demo

The [large-data companion](pod_demo_eeg_large.ipynb) adds the complete
Alljoined-1.6M scheduled corpus and actual disk encoding/reconstruction. See
[the large-corpus workflow](#complete-alljoined-16m-corpus) below.

[`pod_demo_eeg.ipynb`](pod_demo_eeg.ipynb) applies SVD-ROM's POD to every sample
of the 13 continuous Alljoined-1 EEG recordings. It conditions slow drift,
centers each recording, fits channel modes on 12 complete recordings, and
reconstructs subject 08 using frozen modes. It also excludes each of the eight
participants in turn, including all of that participant's sessions. The FIF
files total about 5.9 GB. Processing uses bounded time blocks, with no stimulus
images or labels required.

## Setup

From the repository root, install the demo dependencies with `uv`:

```bash
uv sync --locked --extra demos --extra neuro
```

Acquire the Alljoined-1 EEG recording from the
[original OSF release](https://osf.io/kqgs8/) (also indexed by
[NEMAR](https://ww2.nemar.org/dataset/nm000133)). Place all 13 continuous
`subj*_session*_eeg.fif` files under `demos/data/alljoined1/`. This is the
directory containing `subj01_session1_eeg.fif` and
`subj08_session1_eeg.fif`. You can instead set `SVDROM_EEG_DIR` to the directory
containing those files before starting Jupyter:

Use the original continuous FIF exports rather than `*_epo.fif` epochs or the
BDF layout of a newer mirror. The mirror's directory layout can differ from the
OSF release expected by this notebook.

```bash
export SVDROM_EEG_DIR=/path/to/alljoined1/raw
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd demos
../.venv/bin/jupyter lab pod_demo_eeg.ipynb
```

The data are not tracked by this repository. Waveform previews use short
windows; full-recording spectra and diagnostics use chunkwise reductions. The
associated NSD images are not required. Fitting all
recordings and scoring the full held-out recording can take several minutes
on a laptop. The Dask worker count and chunk length can be adjusted in the
notebook if memory is limited.

The notebook produces original-signal audits, waveform reconstructions, channel
modes, schematic sensor maps, whole-recording power spectra, band-power errors,
correlation matrices, rank/compression curves, channel errors and participant
validation plots. The source FIF files do not contain measured electrode
positions, so sensor maps use a labelled standard template.

The reconstruction target is the conditioned within-recording fluctuation
signal, evaluated against a zero-fluctuation baseline. This is batch compression:
means from the encoded recording are stored metadata, while its participant is
excluded from learning the spatial basis. Nominal float32 storage ratios count
modes, coefficients and means, excluding container and coordinate overhead.
Decoded waveforms use float32 modes and coefficients, and their full-recording
errors are checked against the float64 covariance projection. Spectral
summaries are combined in a reduction tree, avoiding one retained result array
per block as recording duration grows.
Conditioning intentionally discards slow variation; these scores do not claim
recovery of the unfiltered source.

## Resource benchmark

From the repository root, with `SVDROM_EEG_DIR` set:

```bash
uv run --locked --extra demos --extra neuro python -m demos.benchmark_eeg
```

The default run benchmarks 1, 4, and 12 training recordings sequentially, each
in a fresh worker process, including filtering, estimation of recording means,
fitting and whole held-out reconstruction/error scoring. It
writes `demos/eeg_benchmark.json`, which the notebook plots. Use `--counts`,
`--chunk-samples`, `--workers`, and `--output` to adjust the run. Peak process
RSS is measured on Linux/macOS; other platforms can run the notebook but the
benchmark's memory measurement currently requires one of those systems.
Workers use two Dask threads and one BLAS thread by default, with no GPU.

The checked-in report records package versions, platform, environment-visible RAM,
file sizes, sample counts, channel order, and calibration/reference metadata.
It contains a reference run, not measurements from whoever opens the notebook.
The file manifest is not a cryptographic integrity check. Each worker's peak
includes all worker stages and rank-eight held-out evaluation, but excludes the
parent process, notebook plots, Welch analysis, participant validation, other
applications and filesystem cache. The report records conditioning settings;
the notebook rejects a report from the previous preprocessing configuration.
Timing is sensitive to other workloads and warm filesystem caches. The signal
matrix here is smaller than the RAM visible in the measured WSL2 environment,
so this demonstrates process memory well below input size. The report does not
independently establish the physical laptop RAM.

## Signal-quality audit

The notebook checks that FIF channel units are volts before converting MNE's
calibrated data to microvolts. It reports channel offsets and standard deviations
for every recording, the contribution of differences between session means to
pooled raw training energy, and the fraction of source variance retained after
conditioning. The original pooled fit captured 94.8% of training energy with
eight modes, but 97% of that pooled fluctuation energy came from differences
between session means. The revised model targets within-recording fluctuations.

The [authors' preprocessing workflow](https://github.com/Alljoined/alljoined-dataset1/tree/main/1_preprocessing)
distinguishes continuous FIF exports from cleaned, filtered epochs. This demo
uses the continuous exports, applying a 0.5 Hz fourth-order Butterworth high-pass
forward and backward, followed by per-recording centering. Forward/reverse
filtering has zero phase and squares the single-pass amplitude response. Tasks
read 32 seconds of real neighboring signal on each side and discard these halos
after filtering; recording boundaries use filter padding. Tests compare blocked
and whole-recording filtering to check that seams do not introduce appreciable
errors. No channels or samples are removed and the input reference is retained.
This lightweight conditioning does not reproduce the authors' full cleaning
pipeline. Header calibration and reference flags cannot
independently establish hardware calibration or the physical acquisition
reference. Large offsets and drift therefore remain explicit limitations for
physiological interpretation. Line noise, ocular activity and electrode artifacts
can still influence the modes.

## Statistical checks and results

Whole-recording covariance determines analytical linear projection MSE and channel
correlations. Cached recording covariances reproduce participant validation
without eight repeated full-corpus disk scans. The small eigenproblem is checked
against SVD-ROM's actual fit, and full waveform errors are also computed through
the public projection interface for ranks 8 and 32.

Welch spectra use every complete eight-second window of subject 08 with 50%
overlap and alignment preserved across task boundaries. Band-power errors cover
delta, theta, alpha, beta, 30-45 Hz and both 50/60 Hz line-noise bands.
Correlation RMSE excludes diagonal entries and constant channels. Participant
summaries give equal-participant averages and observed ranges. The
participant-fitted optimum is a descriptive upper bound, not a held-out model.
These are exploratory checks: each participant is excluded from basis fitting,
but the workflow was developed after inspecting the dataset. The results are
not blind confirmatory validation. Spectra in the smaller demo use the float32
basis projection; unlike waveform checks, they do not separately quantize the
coefficients. The large demo computes spectra from actual disk factors.

Execution writes `demos/eeg_validation.json` with package versions, the input
manifest, conditioning settings, reconstruction metrics, spectral errors and
participant fold scores. The file manifest is not a cryptographic integrity check.
The scientific structure parallels airfoil POD's fluctuation analysis, ERA5
SVD's encoding/reconstruction, and the baseline evaluation in weather DMD.

In the reference run, subject 08 retains 81.3% fluctuation energy at eight modes
(about 8x nominal compression) and 94.8% at 32 modes (about 2x). Median absolute
alpha/beta power errors are about 28%/31% and 7%/8%, respectively. Across all
eight held-out participants, energy retention averages 61.2% at eight modes and
90.8% at 32 modes, with observed ranges 5.8%-81.3% and 77.7%-96.0%.

The channel audit identifies an extreme imbalance in subject 03/session 1:
PO4 contributes 99.87% of that conditioned recording's variance (standard
deviation about 3963 microvolts). It is retained and flagged explicitly.
The poor eight-mode transfer to that participant is consistent with this
imbalance, which also qualifies the pooled training energy score. This does not
establish the acquisition cause. Acquisition-aware investigation is
needed before physiological interpretation; declared bad-channel lists,
conditioning and larger rank do not establish neural signal quality.

## Complete Alljoined-1.6M corpus

[`pod_demo_eeg_large.ipynb`](pod_demo_eeg_large.ipynb) extends the demonstration
to the [authors' Alljoined-1.6M release](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M).
This is a different acquisition: 32 electrodes at 256 Hz. All 20 participants,
four sessions per person and 19 blocks per session are included: 1,520 scheduled
continuous EDF recordings. The audited float32 EEG matrix is 14.18 GiB and
contains 129 hours of signal. Participants 1-19 train the shared basis and
participant 20 is held out; all 20 are encoded and decoded. Images, annotations
and telemetry are not compression targets. No stimulus decoding is performed.

The source directory must contain `sub-01/session_01/block_01/` through
`sub-20/session_04/block_19/`. Set `SVDROM_EEG_LARGE_DIR` to the **raw_eeg**
directory, not the dataset parent. EEGVIX already has this complete scheduled
corpus under `challenge/runtime/data/Xu2025Alljoined/download/raw_eeg`; the demo
can read it in place. It does not modify that project's inputs or caches.
For another machine, use the authors' download instructions and acquire only
the EEG files needed here; stimuli and preprocessed image-response tensors
are unnecessary.

The audit validates each EDF header, calibration endpoints, selected electrode
names, native sample rate and payload length. Participant 8's known subject-19
filename errors are handled using the directory identity, following NeuralFetch.
Noncanonical EDF paths are listed separately rather than added to the scheduled
corpus. The local reference download has 61 such paths. Size/mtime metadata and
payload length checks do not constitute cryptographic integrity verification.

### Chunked storage and processing

The pipeline uses [chunked HDF5 arrays](https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage)
and [lazy Dask arrays](https://docs.dask.org/en/stable/array-creation.html).
Parquet can support partial reads of tables; HDF5 directly represents these
channel-by-time matrices and is already a demo dependency. The memory savings
come from bounded reads, lazy reductions and incremental writes. Changing the
file format alone does not make eager full-array computation memory efficient.

Preparation reads 16,384-sample blocks with 32-second filter halos, applies the
same 0.5 Hz high-pass as the smaller demo, and writes float32 signal blocks.
It then centers each recording in a separate bounded pass and stores its mean
and 32x32 covariance. The cache can resume completed records; changed provenance
or settings require a new cache path. A covariance POD fit reduces the lazy
matrix to a 32x32 eigenproblem. Time coefficients remain lazy during fitting.
No downsampling or electrode/recording rejection is used.
The covariance backend stores a square channel matrix and squares the condition
number; weak modes are less accurate than with direct SVD. Its suitability here
depends on the small spatial dimension, not a universal memory bound.

The encoding stage calls the public POD projection interface for every block,
then writes float32 coefficients and a shared float32 basis at ranks 8 and 16.
Training-length normalization is absorbed into the coefficients, so the saved
decoder is `modes @ coefficients`. Per-record filtered means are stored as
metadata. Verification reopens those files and decodes every sample from the
saved factors. Errors are relative to the conditioned signal; this does not
recover slow variation discarded during conditioning.
The notebook checks factor manifests, conditioning, training participant IDs,
rank, dtypes and recording dimensions. Saved spectra identify their source
manifest, participant, Welch window length, basis fingerprints and encoded file
sizes, and are rejected if these disagree with current factors. Basis fingerprints
do not checksum every coefficient; full decoded verification remains separate.

Both nominal factor payload bytes and actual file sizes are reported, including
HDF5 chunk padding, record metadata and container overhead in the latter.
No generic lossless compression filter is enabled. For 32 channels the nominal
rank-eight/rank-sixteen ratios are approximately 4x/2x. EDF file sizes also
include equipment telemetry and 16-bit storage, so comparisons to EDF bytes
must be distinguished from comparisons to the float32 analysis matrix.

### Run the full measurement

From the repository root, using the same demo dependencies as above:

```bash
export SVDROM_EEG_LARGE_DIR=/path/to/Alljoined-1.6M/raw_eeg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run --locked --extra demos --extra neuro python -m demos.benchmark_eeg_large
```

Defaults use two Dask threads, one BLAS thread and no GPU. Preparation is serial.
The command measures preparation, fresh-process fits on 1/5/10/19 participants,
and a separate fit/encode/verify worker. Complete-subject Welch spectra use all
complete eight-second windows with 50% overlap, keeping recording boundaries
separate. Leave-one-participant-out results for all 20 participants reuse the
cached covariances; these are separate from the frozen participant-20 decoder.

Generated arrays live in ignored `demos/data/alljoined16/`: `conditioned.h5`,
`encoded/rank08.h5`, `encoded/rank16.h5`, and `encoded/spectra.h5`. Allow about
30 GiB of additional disk space for these artifacts; the source data remain in
place. Encodings are written exclusively to new files. To repeat a benchmark,
reuse the prepared cache and pass a fresh `--encoded-dir` and `--output`.
Preparation reports identify how many records were reused. A saved preparation
worker result can be supplied with `--preparation-report` to preserve its
original measured cost instead of timing cache reuse.

Open `pod_demo_eeg_large.ipynb` from `demos/` after the benchmark. Set
`SVDROM_EEG_LARGE_CACHE`, `SVDROM_EEG_LARGE_REPORT`, and
`SVDROM_EEG_LARGE_ENCODED` if using custom cache, report, or encoded-directory
paths. In particular, set `SVDROM_EEG_LARGE_ENCODED` to the directory passed
with `--encoded-dir` when repeating the benchmark into a new directory.
The notebook reports the checked-in reference measurements separately from its
current fit, and plots input versus memory, elapsed time, actual storage versus
reconstruction error, participant validation, spectral errors, saved-factor
waveforms and channel quality. Whole-worker RSS includes the worker's imports
and stages, excluding the notebook, other processes and filesystem cache.
Timings depend on shared workloads and cache state. Exceeding RAM is assessed
against the RAM visible inside the reference WSL2 environment, not independently
measured physical laptop RAM or every commercial laptop's capacity. Process RSS
does not include total system memory, and Dask graph metadata can grow as more
blocks are scheduled. The measurements establish the observed range.

The reference run fitted the 13.49 GiB training matrix in 49.6 seconds at
0.41 GiB process peak RSS, against 7.39 GiB RAM visible inside WSL2. Complete preparation
took 9.6 minutes at 0.35 GiB peak; the fit/encode/decode/spectral worker peaked
at 0.43 GiB. These are measurements of different stages, not an end-to-end
49.6-second runtime.

Actual rank-eight/rank-sixteen files occupy 3.80/7.58 GiB, giving 3.74x/1.87x
ratios against the 14.18 GiB float32 EEG matrix. Saved-factor reconstruction
retains 94.1%/98.1% of participant 20's conditioned energy, with RMSE
4.83/2.76 microvolts. Median absolute alpha/beta band-power errors are
27.8%/23.9% and 5.4%/6.3%. Across the 20 participant validation folds,
equal-participant energy retention averages 92.1%/96.8%, ranging
81.6%-99.4% and 89.7%-99.8%. These results are from a different acquisition
and rank-to-channel fraction than the 64-channel demo; they do not isolate
the effect of adding data. The channel audit flags a recording in participant
13 where Fp1 contributes 89.3% of conditioned variance; it remains included.


