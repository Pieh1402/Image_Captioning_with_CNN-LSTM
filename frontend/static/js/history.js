const DemoHistory = (() => {
  const key = "caption_demo_history";
  const limit = 20;

  function load() {
    try {
      const items = JSON.parse(localStorage.getItem(key) || "[]");
      return Array.isArray(items) ? items : [];
    } catch {
      return [];
    }
  }

  function save(items) {
    localStorage.setItem(key, JSON.stringify(items.slice(0, limit)));
  }

  function add(item) {
    const current = load().filter((entry) => entry.image_url !== item.image_url);
    save([item, ...current]);
  }

  function clear() {
    localStorage.removeItem(key);
  }

  return { load, save, add, clear };
})();
