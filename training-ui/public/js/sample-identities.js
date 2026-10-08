/* Filename metadata shared by the Samples UI and API. No PNG or prompt reads. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AnimaSampleIdentity = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function parse(image) {
    const name = typeof image === "string" ? image : image.name || "";
    const dir = typeof image === "string" ? "" : image.dir || "";
    const native = name.match(/^(?:(.*?)__)?prompt_(\d+)_seed_(-?\d+)\.(?:png|jpe?g|webp)$/i);
    const legacy = name.match(/(?:^|_)(e?\d{6,})_(\d{2,})_(\d{14})(?:_|\.)/);
    const legacyGroup = name.match(/_(\d{2,})_\d{14}(?:_|\.)/);
    let promptIndex = null;
    let step = null;
    let epoch = null;
    let checkpoint = null;
    let preview = false;
    let seed = null;
    if (native && Number(native[2]) > 0) {
      // Native 3.8B filenames number prompts from 1; the gallery uses 0.
      promptIndex = Number(native[2]) - 1;
      seed = native[3];
      const prefix = native[1] || "";
      const trained = prefix.match(/-step(\d+)$/);
      const trial = prefix.match(/__preview_step_(\d+)$/);
      if (trained) {
        step = Number(trained[1]);
        checkpoint = native[1] + ".safetensors";
      } else if (trial) {
        step = Number(trial[1]);
        preview = true;
      } else {
        const folder = dir.match(/(?:^|[\\/])step_(\d+)(?:[\\/]|$)/);
        if (folder) step = Number(folder[1]);
      }
    } else if (legacy) {
      promptIndex = Number(legacy[2]);
      if (legacy[1].startsWith("e")) epoch = Number(legacy[1].slice(1));
      else step = Number(legacy[1]);
    } else if (legacyGroup) {
      // Original manual generations do not necessarily include a training step.
      promptIndex = Number(legacyGroup[1]);
    }
    return { promptIndex, step, epoch, checkpoint, preview, seed,
      groupKey: promptIndex === null ? "default" : String(promptIndex) };
  }
  function compareNewest(a, b) {
    const left = parse(a);
    const right = parse(b);
    if (left.step !== null && right.step !== null && left.step !== right.step)
      return right.step - left.step;
    if (left.epoch !== null && right.epoch !== null && left.epoch !== right.epoch)
      return right.epoch - left.epoch;
    return (b.mtime || 0) - (a.mtime || 0);
  }
  function caption(image) {
    const identity = parse(image);
    if (identity.step !== null) return `${identity.preview ? "Preview · " : ""}Step ${identity.step}`;
    if (identity.epoch !== null) return `Epoch ${identity.epoch}`;
    return "";
  }
  return { parse, compareNewest, caption };
});
