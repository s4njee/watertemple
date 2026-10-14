# WebGPU support matrix (story 1.10)

**Checked on 2026-10-08.** Feeds story 12.6 (compatibility notes) and the "optional features required" column of the 2.6 scorecard.

How to read this file

- Source tags like `[S3]` point to the numbered list at the bottom. A suffix names the sub-page, file, or version inside that source (for example `[S4: float32-filterable]` is the per-feature page on web3dsurvey.com, `[S5: GPU]` is `api/GPU.json` in MDN BCD, `[S29: 147]` is the Firefox 147 release notes).
- "unverified" means I could not confirm it from a primary or otherwise reliable source this session. "Conflict" means two sources disagree and both are given.
- web3dsurvey percentages are **share of survey reports**, not share of users. For feature pages they are "among reports tested for that feature". The site shows no date stamp on its pages, so treat the numbers as a snapshot taken today [S4].
- Version snapshot today: Chrome 155 stable (released 2026-10-06; 156 is beta, stable 2026-10-20) [S40]; Firefox 157.0.1 (released 2026-09-29; 158 is due 2026-10-09) [S41]; Safari 27.0 (WebKit's "features" post is dated 2026-09-17) [S26]; Samsung Internet 30 = Chromium 143 [S45].

---

## 1. Browser and platform support

### 1.1 Matrix

| Browser | Windows | macOS | Linux | ChromeOS | Android | iOS / iPadOS |
|---|---|---|---|---|---|---|
| **Chrome** | **113** (needs Direct3D 12) [S3][S6]. Windows ARM64: behind `chrome://flags/#enable-unsafe-webgpu` [S3]. | **113** [S3]. Dawn requires macOS 13+ as of Chrome 153-154 [S18]. | **144**: Intel Gen12+ only [S15]. **147**: NVIDIA (driver 535.183.01+) on Wayland [S3][S17]. Everything else (incl. AMD): behind flags [S3]. Caniuse: "depends on hardware and drivers" [S2]. | **113**, devices with Vulkan [S6][S44]. | **121**, Android 12+ with Qualcomm and ARM GPUs [S7]; the wiki also lists Intel [S3]. Imagination GPUs: **139**, Android 16+. Samsung Xclipse and others: "TBD" [S3]. | n/a (status of third-party iOS browsers not checked here) |
| **Edge** | **113** [S1][S2] | **113** [S1][S2] | Follows Chromium. Edge-specific rollout unverified. | Not applicable (unverified) | Follows Chromium. Edge-specific rollout unverified. | n/a |
| **Safari** | n/a | **26**, default-on only on **macOS 26 Tahoe+**. Safari 26/27 on older macOS needs the `WebGPU` feature flag [S2 notes 2 and 7][S3]. | n/a | n/a | n/a | **26** (iOS 26 / iPadOS 26) on by default [S3][S24]. iOS 17.4-18.7: feature flag only, off by default [S2]. |
| **Firefox** | **141**, all contexts except service workers [S29: 141][S30] | Apple Silicon: **145** on macOS 26+, **147** on all macOS versions [S29: 147][S5: GPU]. Intel Macs: not shipped (Nightly only) [S3][S5: GPU]. | Nightly only [S3]. A user report says the Firefox 152 release channel works behind `dom.webgpu.enabled` in `about:config` [S31] (single report, not a Mozilla statement). Mozilla "expects to ship in 2026" [S3]. | n/a | Behind a flag, Beta/Nightly only (`gfx.webgpu.ignore-blocklist`) [S3]. Caniuse says `dom.webgpu.enabled` [S2]. **Conflict** on the pref name. | n/a |
| **Samsung Internet** | n/a | n/a | n/a | n/a | Caniuse says **24** [S2]. But Samsung 24 is Chromium 117 and Android WebGPU first shipped in Chromium 121 = Samsung **25** [S45][S7]. **Conflict**; treat 25+ as the safe floor. Samsung 30 = Chromium 143, about 12 milestones behind Chrome 155, so it lacks Chrome 144+ items such as compat mode (146) and the per-stage storage limits (146) [S45][S40]. | n/a |

Other rows worth knowing: Opera 99+ (desktop), Opera Mobile 80+ [S2]. UC, QQ, Baidu, KaiOS, Opera Mini: no [S2]. Safari on visionOS 26 [S3].

### 1.2 Caveats that decide "adapter or null"

- `navigator.gpu` existing does not mean an adapter is available. Chrome returns `null` from `requestAdapter()` when the GPU is blocklisted (`chrome://gpu` shows "WebGPU has been disabled via blocklist or the command line"), when "Use graphics acceleration when available" is off, or when no GPU is found [S20].
- Chrome on Windows uses a single adapter at a time and `powerPreference` has no effect there [S20].
- `chrome://gpu` can show "WebGPU: Software only, hardware acceleration unavailable" [S20]. Detect software adapters with `adapter.info.isFallbackAdapter` (Chromium sets it for CPU-type adapters) [S34][S5: GPUAdapterInfo]. Only 0.05% of working survey adapters were fallback [S4].
- Secure context only (HTTPS or localhost) [S20].
- GPU and OS blocklists for Safari and Firefox: **unverified**. Firefox has a blocklist pref (`gfx.webgpu.ignore-blocklist`) but I could not find its contents [S3].

### 1.3 Share of users

| Measure | Value | Source |
|---|---|---|
| caniuse global usage, full support | **85.8%** (+2.02% partial = 87.82%), as returned by the live page today | [S1] |
| caniuse JSON snapshot in the repo | 85.72% full, 3.05% partial (older snapshot, differs slightly) | [S2] |
| web3dsurvey, all reports with a working adapter | **82.71%** | [S4] |
| web3dsurvey by platform | ChromeOS 92.7%, macOS 90%, Windows 87.48%, iOS 85.26%, Android 73.81%, **Linux 17.46%** | [S4] |
| web3dsurvey by browser | Edge 93.09%, Opera 88.97%, Chrome 85.01%, Safari 80.38%, Samsung Internet 77.32%, **Firefox 60.39%** | [S4] |

Read: roughly 15-17% of visitors have no WebGPU adapter today. That is large enough that the poster/video fallback is a primary path, not an edge case. Both figures are survey/market-share approximations, not measurements of our audience.

---

## 2. Optional features

### 2.1 Which browsers expose each feature

Versions are first stable release on Windows/macOS/ChromeOS per MDN BCD unless noted [S5: GPUSupportedFeatures]. Chrome Android is 121 or the same version as desktop, whichever is later [S5]. Linux (Intel Gen12+) follows from 144 [S15]. Edge mirrors Chrome.

| Feature | Chrome / Edge | Safari (macOS 26+, iOS 26+) | Firefox | Notes |
|---|---|---|---|---|
| `timestamp-query` | **121**. Timestamps are quantized to 100 microseconds unless the "WebGPU Developer Features" flag is on [S7] | **26** [S5]. WebKit only exposes it if Metal offers a timestamp counter set (trunk source) [S33] | Implemented in source [S32]; survey 100% [S4]; **Conflict**: MDN BCD says no [S5] (appears stale) | Counter resets can give negative deltas [S7] |
| `float32-filterable` | **119** (Android 121) [S5] | **Conflict**: BCD says no [S5]; survey shows 57.7% of Safari and 55.0% of iOS reports have it [S4: float32-filterable]. Trunk source gates it on `supports32BitFloatFiltering` [S33], i.e. per-device | Implemented in source [S32]; survey 99.5% | Without it, `r32float`/`rg32float`/`rgba32float` are `unfilterable-float` only [S22] |
| `float32-blendable` | **132** [S5]. Since 134, filterable no longer implies blendable [S21: 134] | **26** [S5] | Firefox **158** (partial, per BCD; 158 releases 2026-10-09) [S5]; survey today 3.3% [S4] | Independent of filterable |
| `shader-f16` | **120** (Android 121) [S5]. Linux NVIDIA gets it in 155-156 [S19] | **26** [S5] | Implemented in source [S32]; survey 92.8% [S4]; BCD says no (stale) | Needs `enable f16;` in WGSL [S22] |
| `bgra8unorm-storage` | **113** (Android 121) [S5]. Read-only bgra8unorm storage was removed in 143 [S14] | **26** [S5] | Implemented in source [S32]; survey 99.0% | Only write-only storage for this format [S22] |
| `rg11b10ufloat-renderable` | **113** [S5] | **26** [S5] | Implemented in source [S32] | Also implied by `texture-formats-tier1` [S22] |
| `depth32float-stencil8` | **113** [S5] | **26** [S5] | Implemented in source [S32] | |
| `dual-source-blending` | **130** [S21: 130] | No (BCD [S5]; Safari 27.0 notes list only `clip_distances`) [S26]; survey iOS 0.06% | **155**, desktop [S29: 155] | |
| `clip-distances` | **131** [S21: 131] | **27.0** [S26][S27]. BCD still says no (**conflict**, BCD stale). Survey: Safari 31.4%, iOS 33.7% and rising [S4] | Implemented in source [S32] but survey 1.3%, so not in a release yet; release version unverified | Safari 27.0 shipped 2026-09-17 |
| `subgroups` | **134** [S5]; `subgroup_id` in 144 [S15]; `subgroup-size-control` in 151-152 [S21] | Not shipped. BCD marks `subgroupMinSize/MaxSize` as "preview" [S5: GPUAdapterInfo]; survey 0.3% [S4]. Trunk source enables it on Metal 3 (not released) [S33] | **Not implemented** (Mozilla bug 1955417) [S32] | Never assume it |
| `texture-compression-bc` | **113** [S5] | Per-device: survey 56.3% Safari, 53.8% iOS [S4]; BCD says no [S5]; trunk gates on `supportsBCTextureCompression` [S33] | Implemented in source [S32]; survey 99.6% | Desktop-class GPUs |
| `texture-compression-etc2` | **113** [S5] | **26** [S5]; survey 99.3% [S4] | Implemented in source; survey 24.6% (Apple Silicon only) [S4] | Mobile and Apple GPUs |
| `texture-compression-astc` | **113** (Android 121) [S5] | **26** [S5]; survey 99.4% [S4] | Implemented in source; survey 24.3% | Windows reports 0.2% [S4] |
| `texture-formats-tier1` | **142** [S13][S5]. JS-visible in `wgslLanguageFeatures` only from 153 [S18] | **26.2** per BCD [S5]; **not mentioned** in WebKit's 26.2 notes [S25]; survey 97.8% [S4] | **Not implemented** (absent from Firefox's feature switch and IDL) [S32] | See section 4 |
| `texture-formats-tier2` | **142** [S13][S5] | Partial: survey 30.9% Safari, 33.4% iOS [S4]. Trunk gates tier1/tier2 on ARM64 [S33]. BCD: no | Not implemented [S32] | Read-write storage textures |
| `core-features-and-limits` | **139** [S12] | **26** [S5] | Always set [S32] | Absent only on compat-defaulting adapters [S22] |

### 2.2 Share of devices (web3dsurvey, % of reports tested for the feature)

Source for every row is the per-feature page, `[S4: <feature>]`. Platform columns are by OS across all browsers; browser columns across all OSes.

| Feature | All | Android | iOS | macOS | Windows | Linux | ChromeOS | Chrome | Safari | Firefox | Samsung |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `timestamp-query` | 98.8 | 97.8 | 100 | 96.0 | 99.9 | 99.3 | 100 | 99.4 | 93.1 | 100 | 98.3 |
| `float32-filterable` | 91.1 | 70.4 | 55.0 | 87.4 | 100 | 91.6 | 100 | 94.9 | 57.7 | 99.5 | 75.9 |
| `float32-blendable` | 89.1 | 56.8 | 100 | 94.7 | 91.8 | 75.6 | 81.9 | 92.4 | 100 | 3.3 | 59.3 |
| `shader-f16` | 94.0 | 86.2 | 100 | 99.8 | 91.0 | 74.2 | 99.8 | 93.3 | 100 | 92.8 | 93.7 |
| `bgra8unorm-storage` | 96.1 | 69.4 | 100 | 100 | 97.3 | 89.4 | 100 | 95.1 | 100 | 99.0 | 73.6 |
| `rg11b10ufloat-renderable` | 100 | 99.8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `depth32float-stencil8` | 100 | 99.8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `dual-source-blending` | 82.4 | 69.5 | 0.1 | 68.5 | 99.2 | 84.4 | 99.8 | 93.9 | 0.1 | 93.2 | 69.4 |
| `clip-distances` | 78.9 | 50.9 | 33.7 | 74.6 | 91.7 | 74.7 | 81.9 | 89.8 | 31.4 | 1.3 | 58.6 |
| `subgroups` | 68.4 | 82.2 | 0.1 | 62.6 | 77.7 | 77.5 | 72.9 | 83.1 | 0.3 | 0.0 | 87.9 |
| `texture-compression-bc` | 86.9 | 35.1 | 53.8 | 87.0 | 100 | 82.1 | 82.3 | 89.3 | 56.3 | 99.6 | 46.5 |
| `texture-compression-etc2` | 47.8 | 99.9 | 100 | 95.5 | 0.2 | 71.4 | 99.9 | 47.1 | 99.3 | 24.6 | 100 |
| `texture-compression-astc` | 47.8 | 99.9 | 100 | 95.4 | 0.2 | 62.7 | 98.6 | 46.9 | 99.4 | 24.3 | 100 |
| `texture-formats-tier1` | 93.2 | 97.8 | 99.1 | 93.3 | 90.9 | 84.6 | 98.9 | 98.6 | 97.8 | 1.2 | 96.5 |
| `texture-formats-tier2` | 82.6 | 97.8 | 33.4 | 71.7 | 89.6 | 84.6 | 98.9 | 95.3 | 30.9 | 0.3 | 96.5 |
| `core-features-and-limits` | 99.1 | 98.5 | 100 | 99.2 | 99.0 | 98.9 | 99.0 | 98.9 | 100 | 100 | 96.5 |

Survey also reports the non-standard `float16-renderable` / `float32-renderable` at about 14% overall, 99.9% on iOS and 99.9% on Safari (Apple-only extensions, not portable) [S4: float32-renderable].

### 2.3 Commonly missing on mobile

Android (Chrome / Samsung Internet), from [S4] per-feature pages and GPU-architecture rows:

| Feature | Android | Where it is missing |
|---|---|---|
| `float32-blendable` | 56.8% | Arm Mali Bifrost 0%, Valhall 13% (Adreno 98-100%) |
| `texture-compression-bc` | 35.1% | Mali 0%, Adreno 6xx 27%, Adreno 7xx 99.9%. Use ASTC/ETC2 on mobile (99.9%) |
| `clip-distances` | 50.9% | Mali 0%; Adreno about 99% |
| `dual-source-blending` | 69.5% | Mali Bifrost 74%, Valhall 41% |
| `bgra8unorm-storage` | 69.4% | Adreno 6xx 0.9%, Adreno 7xx 47.6% |
| `float32-filterable` | 70.4% | Adreno 6xx 9.2%, Adreno 7xx 48.1% (Mali 100%) |
| `shader-f16` | 86.2% | Adreno 6xx 31.3% |
| `subgroups` | 82.2% | Adreno 6xx 16.7% |

iOS / Safari:

| Feature | iOS | Note |
|---|---|---|
| `float32-filterable` | 55.0% | Device-dependent (see 2.1) |
| `clip-distances` | 33.7% | Safari 27.0 just shipped it, so this will climb |
| `dual-source-blending` | 0.06% | Not available |
| `subgroups` | 0.14% | Not available |
| `texture-compression-bc` | 53.8% | Use ASTC (99.95%) |
| `texture-formats-tier2` | 33.4% | tier1 is 99.1% |

Present on essentially all mobile reports (97% or more): `timestamp-query` (Android 97.8%, iOS 100%), `rg11b10ufloat-renderable`, `depth32float-stencil8`, ASTC and ETC2 [S4].

---

## 3. Limits that bite on mobile

Key spec facts [S22]:

- Every adapter must support the **default** limits or better. A **device only gets the default limits unless you request better ones in `requestDevice({ requiredLimits })`**. Reading `adapter.limits` does not change what the device enforces.
- The spec says to request better-than-default limits only when needed, because they can cost performance. User agents may bucket or understate the real hardware limits for privacy; no more than 32 distinguishable configurations are allowed [S22 section 2.2.1].
- Spec ED and the W3C Candidate Recommendation Draft of 2026-09-15 have identical default limits [S22].

Table below: web3dsurvey gives "share of reports whose value is at least X". "Typical" below is the highest value held by most reports on that platform. Source for all real-world columns is `[S4: limits/<name>]`.

| Limit | Spec default (guaranteed) | Compat default | Android (Chrome) | iOS / Safari | Desktop | Bites when |
|---|---|---|---|---|---|---|
| `maxTextureDimension2D` | **8192** | 4096 | 16384 (98%) | 16384 (99%) | 16384 (about 100% Win/ChromeOS, 97-99% Linux/macOS); 32767 on 1-4% of macOS/Linux/Firefox | Almost never; 8192 covers 8K |
| `maxStorageBuffersPerShaderStage` | **8** | 8 (vertex 0, fragment 4) | 10 (100%), 16 (91%) | 10 (99%) | Windows 16 (98%), Linux 16 (92%), ChromeOS 16 (86%); **macOS 16 only 30%** (10 on 94%); Firefox 31+ (76%) | SPH/MPM with many SSBOs |
| `maxStorageTexturesPerShaderStage` | **4** | 4 (vertex 0) | 8 (100%) | 8 (99%) | Windows 8 (100%), macOS 8 (99%); **Linux 8 only 65%; ChromeOS 8 only 19%** (the rest sit at 4); Firefox 16+ (100%) | Multi-field compute sims writing more than 4 images |
| `maxComputeWorkgroupStorageSize` | **16384** B | 16384 | 32768 (94%); **Adreno 6xx only 71%** | 32768 (99%) | 32768 (about 100% Win/macOS); Linux 49152 (63%) | Tiled blur, prefix sums, FFT |
| `maxComputeInvocationsPerWorkgroup` (also `maxComputeWorkgroupSizeX/Y`) | **256** | 128 | **1024 on only 64%**: Mali Bifrost 256, Valhall 1024 on 26%, Adreno 1024 (100%); Samsung Internet 61% | **512** (99%) | Windows 1024 (100%), macOS 1024 (99%), Linux 86%, **ChromeOS 36%** | 32x32 workgroups |
| `maxBufferSize` | **256 MiB** (268435456) | 256 MiB | 1 GiB (100%), 2 GiB-1 (77%) | about 307 MiB (99%), about 614 MiB (95%), 1 GiB (81%) | Windows 2 GiB-4 (100%); macOS about 1.17 GiB (85%) | Large particle or grid buffers |
| `maxStorageBufferBindingSize` (coupled to the above) | **128 MiB** | 128 MiB | 256 MiB (75%), 512 MiB (50%), 1 GiB (36%); **Adreno 7xx: 256 MiB on only 6%** | about 307 MiB (100%), about 614 MiB (90-95%) | Windows 1 GiB (100%) | Single large SSBO |
| `maxColorAttachmentBytesPerSample` | **32** | 32 | 128 (91%); **Samsung Internet 32 only** | **64** (99%) | Windows 128 (100%); macOS 128 (65%, 64 on 99%); Firefox 128 (75%) | MRT/G-buffer: 4x `rgba16float` = 32 B (fits), `rgba32float` costs 16 B each |

Render-target cost per sample (spec table [S22 section 26.1.1]): `rgba8unorm` 8 B, `rgba16float` 8 B, `rgba32float` 16 B, `r32float` 4 B, `rg11b10ufloat` 8 B, `rgb10a2unorm` 8 B.

**Conflict on Apple limits.** web3dsurvey's iOS and Safari rows show 10 storage buffers, 8 storage textures and 512 invocations. Its "apple - apple" GPU-architecture rows (same feature profile as iOS) show 44 / 44 / 1024. WebKit trunk source sets 1024 invocations and derived binding limits [S33]; Safari 27.0 notes mention "restored `maxStorageBuffersInFragmentStage` and related limits" [S26]. I cannot reconcile these. Treat Safari as "default limits plus a little" and request limits only after reading `adapter.limits`.

Newer per-stage limits: `maxStorageBuffersInVertexStage`, `maxStorageBuffersInFragmentStage`, `maxStorageTexturesInVertexStage`, `maxStorageTexturesInFragmentStage` exist in the spec [S22]. Chrome 146, Firefox 151 (partial, reports the per-shader-stage value), Safari 27 [S5: GPUSupportedLimits][S29: 151]. Chrome 146 also lifted `maxStorageBuffersPerShaderStage` up to 16 and `maxSampledTexturesPerShaderStage` up to 48 [S16].

---

## 4. Storage texture formats

Source for this section: WebGPU spec "Plain color formats" table and feature definitions [S22 sections 25.19, 25.20, 26.1.1], WGSL "storage texel formats" [S23]. "WO / RO / RW" = write-only / read-only / read-write access.

### 4.1 What works with no optional feature (core feature level)

| Format | WO | RO | RW | Notes |
|---|---|---|---|---|
| `rgba8unorm` | yes | yes | needs tier2 | Also filterable, blendable, renderable |
| `rgba8snorm` | yes | yes | never | |
| `rgba8uint`, `rgba8sint` | yes | yes | needs tier2 | |
| `rgba16float` | yes | yes | needs tier2 | Filterable, blendable, renderable; multisample needs `core-features-and-limits` |
| `rgba16uint`, `rgba16sint` | yes | yes | needs tier2 | |
| `r32float`, `r32uint`, `r32sint` | yes | yes | **yes** | The only formats with RW in core. `r32float` is unfilterable unless `float32-filterable` |
| `rg32float`, `rg32uint`, `rg32sint` | yes, only with `core-features-and-limits` | same | never | **Not usable in compatibility mode** |
| `rgba32float`, `rgba32uint`, `rgba32sint` | yes | yes | needs tier2 | |
| `bgra8unorm` | needs `bgra8unorm-storage` | no | no | RO was removed in Chrome 143 [S14] |

### 4.2 The ones people ask about

| Format | Core | With `texture-formats-tier1` | With `texture-formats-tier2` |
|---|---|---|---|
| `rgba16float` | WO, RO | same | **adds RW** |
| `r32float` | WO, RO, RW | same | same |
| `rg32float` | WO, RO (not compat) | same | no RW ever |
| `rg16float` | **none** | **adds WO, RO** | no RW |
| `r16float` | **none** | **adds WO, RO** | **adds RW** |

### 4.3 What tier 1 and tier 2 add

- **`texture-formats-tier1`**: enables `rg11b10ufloat-renderable`; adds `r16/rg16/rgba16 unorm and snorm` (render attachment, blendable, multisample, WO and RO storage); render/blend/multisample/resolve for `r8snorm`, `rg8snorm`, `rgba8snorm`; and **WO + RO storage** for `r8*`, `rg8*`, `r16uint/sint/float`, `rg16uint/sint/float`, `rgb10a2uint/unorm`, `rg11b10ufloat`.
- **`texture-formats-tier2`**: implies tier1; adds **RW storage** for `r8unorm/uint/sint`, `rgba8unorm/uint/sint`, `r16uint/sint/float`, `rgba16uint/sint/float`, `rgba32uint/sint/float`.
- WGSL must declare `requires texture_formats_tier1;` to use the tier1 texel formats in `texture_storage_*` types [S23]. RW storage access also relies on the `readonly_and_readwrite_storage_textures` WGSL language feature, shipped in Chrome 124 [S8] and present on 99.0% of survey browsers [S4].
- Detect with `adapter.features.has('texture-formats-tier1')`. Chrome's 142 post uses both "texture-formats-tier1" and "texture-format-tier1" in different places; the spec name is plural [S13][S22]. `wgslLanguageFeatures` only lists `texture_formats_tier1` from Chrome 153 [S18], so do not feature-detect via WGSL on older Chrome.

### 4.4 Where tiers have shipped

| Engine | tier1 | tier2 |
|---|---|---|
| Chrome / Edge | 142 (Oct 2025) [S13][S5] | 142 [S5] |
| Safari | 26.2 per BCD [S5], but not in WebKit's 26.2 notes [S25]; survey 97.8% of Safari reports [S4] | Partial: survey 30.9% Safari / 33.4% iOS [S4]. Trunk source: ARM64 only [S33] |
| Firefox | Not implemented [S32]; survey 1.2% [S4] | Not implemented |
| Survey, Android Chrome | 97.8% | 97.8% |

### 4.5 Compat-mode storage restrictions

`rg32uint/sint/float` cannot be storage textures; vertex stages get 0 storage buffers and 0 storage textures; fragment stages 4 and 4 [S22 limits table][S42].

---

## 5. Related platform APIs

### 5.1 Compatibility mode (`requestAdapter({ featureLevel: "compatibility" })`)

| Engine | Status | Source |
|---|---|---|
| Chrome / Edge | Shipped in **146** (post dated 2026-02-25). BCD marks it "partial": **Android 10+ with OpenGL ES 3.1+** only. Windows (Direct3D 11) and ChromeOS were "being explored" at launch. Origin trial ran Chrome 139-145 | [S16][S5: GPU][S12] |
| Safari | Not supported | [S5: GPU] |
| Firefox | Not supported (the IDL accepts `featureLevel` but BCD says no) | [S5: GPU][S32] |

Rules [S22][S42]:

- If the implementation cannot enforce compat validation it silently returns a **core** adapter. A compat-defaulting adapter does not list `core-features-and-limits`. Request that feature on the device (if the adapter has it) to remove the restrictions.
- Compat default limits: `maxTextureDimension1D/2D` 4096, `maxComputeInvocationsPerWorkgroup` 128, `maxComputeWorkgroupSizeX/Y` 128, `maxColorAttachments` 4, `maxInterStageShaderVariables` 15, `maxUniformBufferBindingSize` 16384, storage buffers/textures in vertex stage 0, in fragment stage 4/4 [S22].
- Other restrictions [S42, a secondary source: spec text for these was not individually re-verified]: one view dimension per texture, no cube arrays, no texture format reinterpretation, no `sample_mask`/`sample_index`, same blend state on all color targets, `rgba16float`/`r32float` not multisampleable, `bgra8unorm-srgb` unsupported.

### 5.2 HTML-in-Canvas (WICG `html-in-canvas`)

**Status: Chromium origin trial / flag only. Not shipped anywhere.**

| Item | Detail | Source |
|---|---|---|
| Chrome | Dev trial behind `chrome://flags/#canvas-draw-element`. **Origin trial** approved for Chrome 148-151 [S38]; the OT post says 148-150 [S35]; the latest update (page last updated 2026-09-29) says the trial is **extended through Chrome 160** [S36]. Tokens must be renewed | [S35][S36][S38] |
| Edge | Own origin trial page; lists the older API names; **trial expiration 2026-10-20** | [S39] |
| Safari (WebKit) | "No signal" in the Intent to Experiment | [S38] |
| Firefox (Gecko) | "No signal" (an earlier draft said "Negative", corrected) | [S38] |
| Spec | Explainer at WICG; HTML spec PR whatwg/html#11588 | [S37][S38] |

API names changed. **Both name sets exist in the wild**:

| Concept | Chrome 148-154 (OT announcement) | Chrome 155+ (per the OT-changes post) |
|---|---|---|
| Opt-in on canvas | `<canvas layoutsubtree>` | `<canvas content="drawable">`, with `drawable` on descendants |
| 2D draw | `drawElementImage()` (returned a transform) | `drawElementImage()` returns void; hit-test sync is handled by the browser |
| WebGL | `texElementImage2D(target, internalFormat, element)` (arguments trimmed in 150) | `texElementSubImage2D`; storage must be allocated first with `texImage2D` |
| WebGPU | `queue.copyElementImageToTexture(element, { texture })` | `queue.drawElementImageToTexture(...)` with `texture`, `size`, `sourceX/Y/Width/Height` |
| Events | `paint` event (`canvas.onpaint`) | same; `requestPaint()`, `captureElementImage()`, `updateElementGeometry()` also appear |

Sources: [S35][S36] for the 148-154 and 155+ columns; the WICG README [S37] already uses the 155+ names. Chrome 155 became stable on 2026-10-06 [S40], so the renamed API is what OT participants on stable will see. Cross-origin iframe content, visited-link state, and several other items are excluded from painting by design [S37][S35]. Three.js has an experimental `THREE.HTMLTexture` using it [S35].

### 5.3 HDR canvas output (`toneMapping: { mode: "extended" }`)

| Engine | Status | Source |
|---|---|---|
| Chrome / Edge | **129** on ChromeOS, macOS, Windows; Linux (Intel Gen12+) from 144; Android mirrors Chrome. Example uses `format: "rgba16float"` | [S10][S5: GPUCanvasContext] |
| Safari | MDN says no [S5]. WebKit implemented the config under bug 279816 (`toneMapping.mode`). In **Aug 2025** a tester found macOS Safari accepted `extended` in `getConfiguration()` but still **displayed SDR**; iOS Safari with the "WebGPU HDR" flag worked [S28]. **2026 status unverified**: Safari 27 notes add `srgb-linear` and `display-p3-linear` predefined canvas color spaces but say nothing about WebGPU tone mapping [S27][S26]. | [S28][S27] |
| Firefox | Not exposed. `toneMapping` and `colorSpace` are commented out in the IDL (`TODO bug 1834395`) | [S32] |

Also in Chrome 155-156: canvas contexts can use `srgb-linear` and `display-p3-linear` for wide-gamut HDR rendering; status "intent to ship" linked, not stated as shipped [S19].

### 5.4 WebGPU in workers with `OffscreenCanvas`

| Engine | Status | Source |
|---|---|---|
| Chrome / Edge | `navigator.gpu` in dedicated workers and `OffscreenCanvas.getContext('webgpu')` from the first release; service and shared workers from **124** | [S5: OffscreenCanvas, WorkerNavigator][S8] |
| Safari | **26** (`WorkerNavigator.gpu` and the OffscreenCanvas WebGPU context) | [S5] |
| Firefox | **141**, all contexts **except service workers** (bug 1942431) | [S5][S29: 141] |

`OffscreenCanvas` itself: Chrome 69, Firefox 105, Safari 16.4 [S5]. Per-engine perf behavior is unmeasured (story 12.7).

### 5.5 `GPUAdapterInfo`

| Topic | Finding | Source |
|---|---|---|
| Availability | `adapter.info` (sync) since Chrome **127**; `device.adapterInfo` since **132**; `requestAdapterInfo()` removed in **131**. Firefox 141 and Safari 26 have `info`. `isFallbackAdapter` moved here in 136 (Chrome); `powerPreference` added in 137 | [S9][S11][S5: GPUAdapterInfo] |
| Spec rule | Every field may be an empty string; "applications must be able to handle any possible `GPUAdapterInfo` values, including the absence of those values" | [S22 section 3.6.2.4] |
| Chrome (stable) | Without the "WebGPU Developer Features" flag the object has only `vendor`, `architecture`, `subgroupMinSize/MaxSize`, `isFallbackAdapter`. **`device` and `description` are empty strings**, populated only with the developer flag. Architecture is reported for recent GPUs (e.g. `blackwell`, `rdna4`, `valhall`) | [S34][S9][S11] |
| Survey view | Vendor and architecture are populated for the large majority of reports (vendors: intel, apple, nvidia, amd, arm, qualcomm, ...) | [S4] |
| Safari | **Unverified for release builds.** WebKit trunk derives vendor, architecture, device and description from the first word of the Metal device name, so they are likely all the same token (for example `apple`) | [S33: GPUAdapterInfo.h] (trunk, not a shipped guarantee) |
| Firefox | Unverified which fields are populated | n/a |
| Privacy | MDN: the information shared is kept to a minimum and different vendors share different types and granularities (fingerprinting risk) | [S43] |
| Unmasking | An old design doc by Toji proposed a permission-gated `requestUnmaskedAdapterInfo()`. It does not appear anywhere in the current spec text, so assume it does not exist | [S46][S22] |

---

## 6. Design implications for this project

1. **Treat spec defaults as the contract for Tier A and B.** Effects that need no `requiredLimits` work on every WebGPU device: 8 storage buffers and 4 storage textures per stage, 16 KiB workgroup memory, 256 invocations per workgroup, 128 MiB storage binding, 256 MiB buffers, 32 B per sample across color attachments, 8192 textures. Request more only per effect, only after reading `adapter.limits`, and record it in the scorecard (2.6). Devices get defaults even if the adapter supports more [S22].
2. **Workgroups: 256 invocations (16x16 or 64x4), 16 KiB shared memory.** Mali Bifrost is 256-only, Safari/iOS reports 512, Samsung Internet and ChromeOS often lack 1024, Adreno 6xx 29% stay at 16 KiB shared memory [S4]. Never default to 1024 or 32 KiB.
3. **Simulation state formats.** Use `rgba16float` (renderable, filterable, blendable, WO/RO storage in core) for velocity/dye/height state, and `r32float` or `r32uint` where RW storage is needed. Avoid `rg32float`, `r16float` and `rg16float` as storage textures (`rg32float` is not allowed in compat; `r16float`/`rg16float` need tier1). Do **not** assume `texture-formats-tier2` RW on `rgba16float`: Firefox lacks it and Safari has about a third. Default to ping-pong with a sampled read texture and a write-only storage target, which works everywhere [S22].
4. **Float32 textures are not linearly filterable by default.** `float32-filterable` is missing on 45% of iOS and 42% of Safari reports, 91% of Adreno 6xx and about half of Adreno 7xx [S4]. Sample `r32float` with `textureLoad` and bilinear in the shader, or avoid float32 render targets. `float32-blendable` is missing on Mali and Firefox (until 158), so no additive blending into `rgba32float` [S4][S5].
5. **Optional features are enhancements with a fallback, never requirements.** `timestamp-query` powers the HUD only (100 microsecond quantization in Chrome [S7], so average over frames) with a rAF/CPU-time fallback; `shader-f16` needs two shader variants chosen at pipeline-creation time; do not use `subgroups` (Safari and Firefox lack it), `dual-source-blending` or `clip-distances` (absent on Mali; dual-source absent on Safari) in any Tier A effect; ship textures as KTX2/Basis and transcode to whichever of ASTC, ETC2 or BC is exposed, else `rgba8unorm` [S4].
6. **Compatibility mode widens Android reach for fragment-only Tier A effects.** Write Tier A effects within compat limits (no storage resources in vertex stages, no `rg32*` storage, 4096 textures, 128 invocations, same blend on all targets, no cube arrays), request `featureLevel: "compatibility"` first on Chrome 146+, then check for `core-features-and-limits` to upgrade. Safari and Firefox return core adapters anyway; compute-based Tier B and C require a core adapter [S16][S22][S42].
7. **Plan for roughly 15-17% of visitors with no adapter.** Linux (17% have WebGPU), Firefox Android, Safari or macOS older than 26, and blocklisted GPUs are all "no". `navigator.gpu` being defined is not enough: null-check `requestAdapter()`, treat `info.isFallbackAdapter` as no-WebGPU for Tier B and C, and keep the poster/video path first-class (story 1.8, 12.5) [S1][S4][S20].
8. **Backdrop capture (story 9.1): HTML-in-Canvas is Chrome-only and unstable.** It is an origin trial with different API names before and after Chrome 155, extended to Chrome 160; Safari and Firefox have no signal. Use it only as progressive enhancement, feature-detect both name sets, and make image/video/`createImageBitmap` the baseline (`copyExternalImageToTexture` with HTMLImageElement/ImageData sources is unsupported in Safari and Firefox per BCD; use ImageBitmap) [S36][S38][S5: GPUQueue].
9. **HDR (story 3.10): render into an offscreen `rgba16float` chain and tone-map to an SDR canvas by default.** `toneMapping: { mode: "extended" }` is Chrome-only; Firefox rejects the option; Safari echoes `extended` from `getConfiguration()` without necessarily displaying HDR (macOS, Aug 2025 report). Gate HDR behind an explicit opt-in and a real-device check [S10][S28][S32].
10. **Defensive plumbing.** Never branch on `adapterInfo.device/description` (empty in stable Chrome); use `adapter.limits/features` snapshots for the scorecard instead. Always pass `GPUTextureView`s to render-pass attachments (Safari 26.2 and 27 only recently fixed passing a bare `GPUTexture` as depth-stencil or resolve attachment). Wire `device.lost` and `uncapturederror` and test Safari below 27, where the `onuncapturederror` handler attribute was broken [S25][S26][S27][S34].

---

## 7. Unverified items and source conflicts

Unverified:

- Edge-specific platform rollout (Linux, Android) and any Edge-specific blocklist.
- GPU/OS blocklists for Safari and Firefox.
- Which `GPUAdapterInfo` fields Safari and Firefox populate in release builds.
- Whether Chrome, Edge or Firefox on iOS inherit Safari's WebGPU status (not checked).
- Safari WebGPU HDR display behavior in 2026 (latest evidence is Aug 2025).
- The end date of the Chrome HTML-in-Canvas origin trial beyond "through Chrome 160".
- Safari 27.0 exact public release date (the features post is dated 2026-09-17).
- Release versions in which Firefox ships `clip-distances`, and when Firefox adds `timestamp-query` etc. (it already works in survey data; the source tree shows them implemented, but only `dual-source-blending` 155, `float32-blendable` 158 and the per-stage limits 151 have release-note citations).

Conflicts:

| Topic | Source A | Source B |
|---|---|---|
| Samsung Internet first version | caniuse: 24 [S2] | Chromium mapping: 25 [S45][S7] |
| Firefox Android enable pref | caniuse: `dom.webgpu.enabled` [S2] | gpuweb wiki: `gfx.webgpu.ignore-blocklist` in Beta/Nightly [S3] |
| Firefox Linux | gpuweb wiki: Nightly only [S3] | GitHub issue: Firefox 152 release works behind `dom.webgpu.enabled` [S31] |
| Firefox optional features | MDN BCD: most "no" [S5] | Firefox source and survey: implemented [S32][S4] |
| Safari `float32-filterable` | MDN BCD: no [S5] | Survey: 55-58% [S4]; trunk source: per-device [S33] |
| Safari `clip-distances` | MDN BCD: no [S5] | WebKit 27.0 notes: supported [S26] |
| Safari `texture-formats-tier1` | MDN BCD: 26.2 [S5] | WebKit's 26.2 notes: not mentioned [S25] |
| Safari limits | Survey iOS/Safari rows: 10 / 8 / 512 | Survey "apple - apple" rows: 44 / 44 / 1024 [S4] |
| caniuse global share | live page 85.8% + 2.02% [S1] | JSON snapshot 85.72% + 3.05% [S2] |
| Linux NVIDIA requirement | Wiki: driver 535.183.01+ on Wayland [S3] | Chrome post: "drivers from 2024-05" on Wayland [S17] |
| Android GPU vendors | Chrome 121 post: Qualcomm and ARM [S7] | Wiki: ARM, Qualcomm and Intel [S3] |
| HTML-in-Canvas names | OT post, Edge OT page, Intent to Experiment: `layoutsubtree`, `copyElementImageToTexture` [S35][S38][S39] | WICG README and Chrome 155 changes: `content="drawable"`, `drawElementImageToTexture` [S36][S37] |
| HTML-in-Canvas OT window | Intent: Chrome 148-151 [S38]; OT post: 148-150 [S35] | Update post: through Chrome 160 [S36] |
| Chrome 142 post feature name | Prose: `texture-formats-tier1` | Code sample: `texture-format-tier1` [S13] (spec uses plural [S22]) |

---

## Sources

Checked 2026-10-08 unless noted.

- [S1] caniuse WebGPU: https://caniuse.com/webgpu
- [S2] caniuse data: https://github.com/Fyrd/caniuse/blob/main/features-json/webgpu.json
- [S3] gpuweb wiki, Implementation Status (page shows "edited Oct 2, 2026"): https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
- [S4] web3dsurvey. Overview: https://web3dsurvey.com/webgpu. Feature pages: `https://web3dsurvey.com/webgpu/features/<name>`. Limit pages: `https://web3dsurvey.com/webgpu/limits/<name>`.
- [S5] MDN browser-compat-data (`api/GPU.json`, `GPUSupportedFeatures.json`, `GPUSupportedLimits.json`, `GPUAdapterInfo.json`, `GPUCanvasContext.json`, `GPUDevice.json`, `GPUQueue.json`, `OffscreenCanvas.json`, `WorkerNavigator.json`): https://github.com/mdn/browser-compat-data/tree/main/api
- [S6] Chrome 113 release: https://developer.chrome.com/blog/webgpu-release
- [S7] What's New in WebGPU, Chrome 121: https://developer.chrome.com/blog/new-in-webgpu-121
- [S8] Chrome 124: https://developer.chrome.com/blog/new-in-webgpu-124
- [S9] Chrome 127: https://developer.chrome.com/blog/new-in-webgpu-127
- [S10] Chrome 129: https://developer.chrome.com/blog/new-in-webgpu-129
- [S11] Chrome 138: https://developer.chrome.com/blog/new-in-webgpu-138
- [S12] Chrome 139: https://developer.chrome.com/blog/new-in-webgpu-139
- [S13] Chrome 142: https://developer.chrome.com/blog/new-in-webgpu-142
- [S14] Chrome 143: https://developer.chrome.com/blog/new-in-webgpu-143
- [S15] Chrome 144: https://developer.chrome.com/blog/new-in-webgpu-144
- [S16] Chrome 146: https://developer.chrome.com/blog/new-in-webgpu-146
- [S17] Chrome 147-148: https://developer.chrome.com/blog/new-in-webgpu-147-148
- [S18] Chrome 153-154: https://developer.chrome.com/blog/new-in-webgpu-153-154
- [S19] Chrome 155-156 (page last updated 2026-10-07): https://developer.chrome.com/blog/new-in-webgpu-155-156
- [S20] Chrome WebGPU troubleshooting: https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips
- [S21] "What's New in WebGPU" index (titles by version, used for 128-134 items): https://developer.chrome.com/docs/web-platform/webgpu/news
- [S22] WebGPU spec, Editor's Draft: https://gpuweb.github.io/gpuweb/ ; W3C Candidate Recommendation Draft 2026-09-15: https://www.w3.org/TR/webgpu/
- [S23] WGSL spec: https://gpuweb.github.io/gpuweb/wgsl/
- [S24] WebKit features in Safari 26.0: https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
- [S25] WebKit features for Safari 26.2 (2025-12-12): https://webkit.org/blog/17640/webkit-features-for-safari-26-2/
- [S26] WebKit features for Safari 27.0 (2026-09-17): https://webkit.org/blog/18325/webkit-features-for-safari-27-0/
- [S27] News from WWDC26, WebKit in Safari 27 beta: https://webkit.org/blog/17967/news-from-wwdc26-webkit-in-safari-27-beta/
- [S28] WebKit bug 272702 (toneMapping): https://bugs.webkit.org/show_bug.cgi?id=272702
- [S29] Firefox release notes for developers: https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/141 (and `/147`, `/151`, `/155`, `/157`)
- [S30] Mozilla GFX blog, "Shipping WebGPU on Windows in Firefox 141": https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/
- [S31] gpuweb issue 6331 (Firefox 152 on Linux): https://github.com/gpuweb/gpuweb/issues/6331
- [S32] Firefox source (main): https://github.com/mozilla-firefox/firefox/blob/main/dom/webgpu/Adapter.cpp and https://github.com/mozilla-firefox/firefox/blob/main/dom/webidl/WebGPU.webidl
- [S33] WebKit source (trunk, not a release guarantee): https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/HardwareCapabilities.mm and https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/WebGPU/GPUAdapterInfo.h
- [S34] Chromium source (main): https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/webgpu/gpu_adapter.cc
- [S35] HTML-in-Canvas origin trial post (page last updated 2026-05-19): https://developer.chrome.com/blog/html-in-canvas-origin-trial
- [S36] HTML-in-Canvas updates (page last updated 2026-09-29): https://developer.chrome.com/blog/html-in-canvas-ot-changes
- [S37] WICG html-in-canvas: https://github.com/WICG/html-in-canvas
- [S38] Intent to Experiment: HTML-in-canvas (2026-04-09): https://groups.google.com/a/chromium.org/g/blink-dev/c/t_nGEmJ_v4s
- [S39] Edge origin trial page: https://developer.microsoft.com/en-us/microsoft-edge/origin-trials/trials/a297467e-0030-4c4c-8739-48e130026c03
- [S40] Chromium milestone schedule: https://chromiumdash.appspot.com/fetch_milestone_schedule?mstone=155 and MDN BCD `browsers/chrome.json`
- [S41] Mozilla product details: https://product-details.mozilla.org/1.0/firefox_versions.json
- [S42] WebGPU Fundamentals, compatibility mode (secondary): https://webgpufundamentals.org/webgpu/lessons/webgpu-compatibility-mode.html
- [S43] MDN GPUAdapterInfo: https://developer.mozilla.org/en-US/docs/Web/API/GPUAdapterInfo
- [S44] Chrome WebGPU overview: https://developer.chrome.com/docs/web-platform/webgpu/overview
- [S45] MDN BCD Samsung Internet release table: https://github.com/mdn/browser-compat-data/blob/main/browsers/samsunginternet_android.json
- [S46] Toji's early adapter-info design doc (old; points readers to the spec): https://hackmd.io/@Toji/S1rvjZD6t
