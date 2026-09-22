(() => {
  const header = document.querySelector("[data-header]");
  const nav = document.querySelector("[data-nav]");
  const navToggle = document.querySelector("[data-nav-toggle]");
  const copyButton = document.querySelector("[data-copy-citation]");
  const copyToast = document.querySelector("[data-copy-toast]");
  const citation = document.querySelector("#citation");

  const updateHeader = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 24);
  };

  const closeNavigation = () => {
    if (!nav || !navToggle) return;
    nav.classList.remove("is-open");
    navToggle.setAttribute("aria-expanded", "false");
    document.body.classList.remove("nav-open");
  };

  navToggle?.addEventListener("click", () => {
    const nextState = navToggle.getAttribute("aria-expanded") !== "true";
    nav?.classList.toggle("is-open", nextState);
    navToggle.setAttribute("aria-expanded", String(nextState));
    document.body.classList.toggle("nav-open", nextState);
  });

  nav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeNavigation);
  });

  window.addEventListener("scroll", updateHeader, { passive: true });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 820) closeNavigation();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNavigation();
  });

  copyButton?.addEventListener("click", async () => {
    const text = citation?.textContent?.trim();
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Copied";
      copyToast?.classList.add("is-visible");
      window.setTimeout(() => {
        copyButton.textContent = "Copy BibTeX";
        copyToast?.classList.remove("is-visible");
      }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      copyButton.textContent = "Selected — press Ctrl+C";
    }
  });

  updateHeader();
})();
