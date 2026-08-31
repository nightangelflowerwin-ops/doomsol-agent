(function () {
  window.installDwasmNormalLoopCollector = function (canvas, options) {
    const rows = [];
    const pending = [];
    const boundaries = [];
    const seed = options.seed || 1;
    const chunkSize = options.chunkSize || 50;
    let flushChain = Promise.resolve();

    window.__teacherStatus = 'running';
    window.__teacherRows = rows;

    const flush = function () {
      if (!pending.length) return flushChain;
      const chunk = pending.splice(0, pending.length);
      const body = chunk.map(row => JSON.stringify(row)).join('\n') + '\n';
      flushChain = flushChain.then(() => fetch('/teacher-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-ndjson' },
        body
      }).then(response => {
        if (!response.ok) throw new Error('teacher output ' + response.status);
      }));
      return flushChain;
    };

    window.__dwasmTeacherSample = function (stateJson, forward, strafe, turn, fire, use, sampleIndex) {
      const state = JSON.parse(stateJson);
      if (!state.ready) return;
      const row = {
        schema_version: 2,
        episode: `e${state.episode}m${state.map}-seed-${seed}`,
        tick: state.tic,
        sample_index: sampleIndex,
        action: { forward, strafe, turn, fire: !!fire, use: !!use },
        state,
        frame: canvas.toDataURL('image/jpeg', 0.86)
      };
      rows.push(row);
      pending.push(row);
      if (pending.length >= chunkSize) flush();
    };
    window.__dwasmTeacherBoundary = kind => boundaries.push({ kind, row: rows.length });
    window.__dwasmTeacherComplete = function () {
      flush().then(() => {
        window.__teacherBoundaries = boundaries;
        window.__teacherStatus = 'complete';
        console.info('Teacher trajectory complete:', rows.length, 'samples');
      }).catch(error => {
        window.__teacherStatus = 'failed';
        window.__teacherError = String(error && error.stack || error);
      });
    };
  };
}());
