const AttentionMap = (() => {
  function normalize(values) {
    const flat = values.flat ? values.flat(Infinity) : values;
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    return values.map((value) => (max === min ? 0.5 : (value - min) / (max - min)));
  }

  function color(value) {
    const ratio = Math.max(0, Math.min(1, value));
    const r = Math.round(40 + ratio * 215);
    const g = Math.round(120 - ratio * 80);
    const b = Math.round(220 - ratio * 190);
    return `rgba(${r}, ${g}, ${b}, ${0.18 + ratio * 0.42})`;
  }

  function draw(canvas, image, weights) {
    if (!canvas || !image || !weights || !weights.length) return;
    const ctx = canvas.getContext("2d");
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const normalized = normalize(weights);
    const side = Math.ceil(Math.sqrt(normalized.length));
    const cellW = width / side;
    const cellH = height / side;
    normalized.forEach((value, index) => {
      const x = (index % side) * cellW;
      const y = Math.floor(index / side) * cellH;
      ctx.fillStyle = color(value);
      ctx.fillRect(x, y, cellW, cellH);
    });
  }

  return { draw };
})();
