(function () {
  function applyLang(lang) {
    document.documentElement.lang = lang;

    var langFields = document.querySelectorAll(".lang-field");
    langFields.forEach(function (el) {
      el.value = lang;
    });

    var textNodes = document.querySelectorAll("[data-i18n-en]");
    textNodes.forEach(function (el) {
      var text = lang === "hi" ? el.dataset.i18nHi : el.dataset.i18nEn;
      if (text) {
        el.textContent = text;
      }
    });

    var placeholders = document.querySelectorAll("[data-placeholder-en]");
    placeholders.forEach(function (el) {
      var text = lang === "hi" ? el.dataset.placeholderHi : el.dataset.placeholderEn;
      if (text) {
        el.setAttribute("placeholder", text);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var select = document.getElementById("lang-select");
    if (!select) return;

    var stored = localStorage.getItem("lang");
    var initial = stored === "hi" ? "hi" : "en";
    select.value = initial;
    applyLang(initial);

    select.addEventListener("change", function () {
      var value = select.value === "hi" ? "hi" : "en";
      localStorage.setItem("lang", value);
      applyLang(value);
    });
  });
})();
