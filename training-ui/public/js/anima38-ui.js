// The native 3.8B backend trains the DiT LoRA and caches frozen text encoders.
function updateAnima38Controls() {
  const native = document.getElementById('cfg-network-module')?.value === 'networks.lora_anima38';
  const te = document.getElementById('cfg-text-encoder-lr');
  const ditOnly = document.getElementById('cfg-unet-only');
  if (te) { te.disabled = native; if (native) te.value = 0; }
  if (ditOnly) { ditOnly.disabled = native; if (native) ditOnly.checked = true; }
  let note = document.getElementById('anima38-mode-note');
  if (!note && te) {
    note = document.createElement('small');
    note.id = 'anima38-mode-note';
    note.textContent = 'Anima 3.8B: both text encoders and the bundled connector stay frozen. Saved trials use Euler.';
    te.parentElement.appendChild(note);
  }
  if (note) note.hidden = !native;
}
document.addEventListener('change', (event) => {
  if (event.target.id === 'cfg-network-module') updateAnima38Controls();
});
