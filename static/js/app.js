(() => {
  const files = { template: null, scene: null };

  const dzTemplate = document.getElementById('dz-template');
  const dzScene = document.getElementById('dz-scene');
  const inTemplate = document.getElementById('in-template');
  const inScene = document.getElementById('in-scene');
  const btnRun = document.getElementById('btn-run');
  const formError = document.getElementById('form-error');
  const processing = document.getElementById('processing');
  const scanImage = document.getElementById('scan-image');
  const stageText = document.getElementById('stage-text');
  const results = document.getElementById('results');
  const statRow = document.getElementById('stat-row');
  const resultImage = document.getElementById('result-image');
  const matchImage = document.getElementById('match-image');
  const objectCards = document.getElementById('object-cards');
  const emptyNote = document.getElementById('empty-note');

  const STAGES = [
    'Extracting SIFT features…',
    'Matching descriptors (ratio test)…',
    'Voting in 4D Hough space…',
    'Fitting affine transforms with RANSAC…',
    'Verifying geometry & suppressing duplicates…',
  ];

  function wireDropzone(zone, input, kind) {
    const preview = zone.querySelector('.dz-preview');
    const removeBtn = zone.querySelector('.dz-remove');

    const setFile = (file) => {
      if (!file || !file.type.startsWith('image/')) return;
      files[kind] = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        zone.classList.add('filled');
      };
      reader.readAsDataURL(file);
      updateRunState();
      hideError();
    };

    zone.addEventListener('click', (e) => {
      if (e.target === removeBtn) return;
      input.click();
    });

    input.addEventListener('change', () => setFile(input.files[0]));

    ['dragenter', 'dragover'].forEach((evt) =>
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
      })
    );
    ['dragleave', 'drop'].forEach((evt) =>
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
      })
    );
    zone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      setFile(file);
    });

    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      files[kind] = null;
      input.value = '';
      preview.src = '';
      zone.classList.remove('filled');
      updateRunState();
    });
  }

  function updateRunState() {
    btnRun.disabled = !(files.template && files.scene);
  }

  function hideError() {
    formError.hidden = true;
    formError.textContent = '';
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function animateCount(el, target, suffix = '') {
    const duration = 650;
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(target * eased) + suffix;
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function buildStatCards(stats) {
    statRow.innerHTML = '';
    const entries = [
      ['Object keypoints', stats.template_keypoints, ''],
      ['Scene keypoints', stats.scene_keypoints, ''],
      ['Tentative matches', stats.tentative_matches, ''],
      ['Detections', stats.detection_count, ''],
    ];
    entries.forEach(([label, value], i) => {
      const card = document.createElement('div');
      card.className = 'stat-card';
      card.style.animationDelay = `${i * 70}ms`;
      const valueEl = document.createElement('div');
      valueEl.className = 'stat-value';
      const labelEl = document.createElement('div');
      labelEl.className = 'stat-label';
      labelEl.textContent = label;
      card.appendChild(valueEl);
      card.appendChild(labelEl);
      statRow.appendChild(card);
      animateCount(valueEl, value);
    });
  }

  function buildObjectCards(detections) {
    objectCards.innerHTML = '';
    detections.forEach((d, i) => {
      const card = document.createElement('div');
      card.className = 'object-card';
      card.style.setProperty('--obj-color', d.color);
      card.style.animationDelay = `${i * 80}ms`;
      card.innerHTML = `
        <h3>Object ${d.id}</h3>
        <div class="metric"><span>Inliers</span><b>${d.inliers}</b></div>
        <div class="metric"><span>Median error</span><b>${d.median_error}px</b></div>
        <div class="metric"><span>Coverage</span><b>${d.coverage}%</b></div>
      `;
      objectCards.appendChild(card);
    });
  }

  async function runDetection() {
    hideError();
    const form = new FormData();
    form.append('template', files.template);
    form.append('scene', files.scene);

    results.hidden = true;
    processing.hidden = false;
    scanImage.src = URL.createObjectURL(files.scene);
    btnRun.classList.add('loading');
    btnRun.disabled = true;

    let stageIndex = 0;
    stageText.textContent = STAGES[0];
    const stageTimer = setInterval(() => {
      stageIndex = (stageIndex + 1) % STAGES.length;
      stageText.textContent = STAGES[stageIndex];
    }, 950);

    try {
      const response = await fetch('/api/detect', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Detection failed.');
      renderResults(data);
    } catch (err) {
      showError(err.message || 'Something went wrong while contacting the server.');
    } finally {
      clearInterval(stageTimer);
      processing.hidden = true;
      btnRun.classList.remove('loading');
      updateRunState();
    }
  }

  function renderResults(data) {
    buildStatCards(data.stats);
    buildObjectCards(data.detections);

    resultImage.src = `data:image/jpeg;base64,${data.detection_image}`;
    matchImage.src = `data:image/jpeg;base64,${data.match_image}`;

    emptyNote.hidden = data.detections.length > 0;

    const frame = resultImage.closest('.image-frame');
    frame.classList.remove('revealed');
    resultImage.onload = () => requestAnimationFrame(() => frame.classList.add('revealed'));

    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  wireDropzone(dzTemplate, inTemplate, 'template');
  wireDropzone(dzScene, inScene, 'scene');
  btnRun.addEventListener('click', runDetection);
})();
