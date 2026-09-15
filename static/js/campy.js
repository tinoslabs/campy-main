/* Campy AI — small progressive-enhancement layer. No framework, no build. */
(function () {
  "use strict";

  /* ---------- Theme ---------- */
  var THEME_KEY = "campy-theme";
  function applyTheme(theme) {
    var resolved = theme;
    if (theme === "system" || !theme) {
      resolved = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    document.documentElement.setAttribute("data-theme", resolved);
  }
  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY) || "system"; } catch (e) { return "system"; }
  }
  window.campyToggleTheme = function () {
    var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
    applyTheme(next);
  };
  applyTheme(storedTheme());

  /* ---------- CSRF ---------- */
  function getCookie(name) {
    var match = document.cookie.match(new RegExp("(^|; )" + name + "=([^;]*)"));
    return match ? decodeURIComponent(match[2]) : null;
  }
  window.campyCsrf = function () { return getCookie("csrftoken"); };

  window.campyFetch = function (url, options) {
    options = options || {};
    options.headers = Object.assign(
      { "X-CSRFToken": getCookie("csrftoken"), "X-Requested-With": "XMLHttpRequest" },
      options.headers || {}
    );
    if (options.json) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
      options.method = options.method || "POST";
      delete options.json;
    }
    options.credentials = "same-origin";
    return fetch(url, options).then(function (response) {
      if (!response.ok) {
        return response.json().catch(function () { return {}; }).then(function (data) {
          throw new Error((data.error && data.error.message) || response.statusText);
        });
      }
      var type = response.headers.get("content-type") || "";
      return type.indexOf("json") > -1 ? response.json() : response.text();
    });
  };

  /* ---------- Dropdowns ---------- */
  document.addEventListener("click", function (event) {
    var trigger = event.target.closest("[data-dropdown]");
    document.querySelectorAll(".dropdown.open").forEach(function (open) {
      if (!trigger || open !== trigger.closest(".dropdown")) open.classList.remove("open");
    });
    if (trigger) {
      event.preventDefault();
      var parent = trigger.closest(".dropdown");
      if (parent) parent.classList.toggle("open");
    }
  });

  /* ---------- Sidebar ---------- */
  document.addEventListener("click", function (event) {
    if (event.target.closest("[data-sidebar-toggle]")) {
      var sidebar = document.querySelector(".sidebar");
      if (sidebar) sidebar.classList.toggle("open");
    }
  });

  /* ---------- Copy to clipboard ---------- */
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-copy]");
    if (!button) return;
    event.preventDefault();
    var value = button.getAttribute("data-copy");
    if (value.charAt(0) === "#") {
      var source = document.querySelector(value);
      value = source ? (source.value || source.textContent) : "";
    }
    var done = function () {
      var original = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = original; }, 1400);
    };
    if (navigator.clipboard) {
      navigator.clipboard.writeText(value).then(done);
    } else {
      var helper = document.createElement("textarea");
      helper.value = value; document.body.appendChild(helper);
      helper.select(); document.execCommand("copy");
      document.body.removeChild(helper); done();
    }
  });

  /* ---------- Confirm before destructive actions ---------- */
  document.addEventListener("submit", function (event) {
    var message = event.target.getAttribute("data-confirm");
    if (message && !window.confirm(message)) event.preventDefault();
  }, true);
  document.addEventListener("click", function (event) {
    var link = event.target.closest("a[data-confirm]");
    if (link && !window.confirm(link.getAttribute("data-confirm"))) event.preventDefault();
  });

  /* ---------- Auto-submit filter forms ---------- */
  document.querySelectorAll("[data-autosubmit]").forEach(function (element) {
    element.addEventListener("change", function () {
      var form = element.closest("form");
      if (form) form.submit();
    });
  });

  /* ---------- Live polling ---------- */
  /* Elements with data-poll="<url>" data-poll-interval="<ms>" refresh their
     own innerHTML from a server-rendered fragment. Chosen over websockets
     deliberately: it survives proxies, needs no extra process, and a control
     room refreshing every few seconds is entirely adequate. */
  function startPolling(element) {
    var url = element.getAttribute("data-poll");
    var interval = parseInt(element.getAttribute("data-poll-interval") || "10000", 10);
    var failures = 0;
    function tick() {
      if (document.hidden) return;            // don't poll a background tab
      fetch(url, { headers: { "X-Requested-With": "XMLHttpRequest" }, credentials: "same-origin" })
        .then(function (response) {
          if (!response.ok) throw new Error(response.statusText);
          return response.text();
        })
        .then(function (html) { element.innerHTML = html; failures = 0; })
        .catch(function () { failures += 1; });
    }
    var timer = setInterval(function () {
      // Back off after repeated failures rather than hammering a dead endpoint.
      if (failures > 5) { clearInterval(timer); return; }
      tick();
    }, interval);
  }
  document.querySelectorAll("[data-poll]").forEach(startPolling);

  /* ---------- Live camera tiles ---------- */
  /* The server holds each camera's stream open and keeps its newest frame, so
     asking for one is a memory read. Tiles poll that rather than each opening
     an MJPEG stream, which on a wall of twenty would pin twenty connections.
     The next frame is decoded off-screen and swapped in, so a tile never
     flickers through a blank. */
  function startTile(img) {
    var url = img.getAttribute("data-live-tile");
    var interval = parseInt(img.getAttribute("data-live-interval") || "1000", 10);
    var inFlight = false;
    var failures = 0;

    function tick() {
      if (document.hidden || inFlight || failures > 8) return;
      inFlight = true;
      var next = new Image();
      next.onload = function () { img.src = next.src; inFlight = false; failures = 0; };
      next.onerror = function () { inFlight = false; failures += 1; };
      next.src = url + "?t=" + Date.now();
    }
    setInterval(tick, interval);
  }
  document.querySelectorAll("[data-live-tile]").forEach(startTile);

  /* ---------- Range value display ---------- */
  document.querySelectorAll("input[type=range][data-output]").forEach(function (input) {
    var output = document.querySelector(input.getAttribute("data-output"));
    if (!output) return;
    var sync = function () { output.textContent = input.value; };
    input.addEventListener("input", sync); sync();
  });

  /* ---------- Tab panels ---------- */
  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    group.querySelectorAll("[data-tab-target]").forEach(function (tab) {
      tab.addEventListener("click", function (event) {
        event.preventDefault();
        var target = tab.getAttribute("data-tab-target");
        group.querySelectorAll("[data-tab-target]").forEach(function (t) { t.classList.remove("active"); });
        tab.classList.add("active");
        document.querySelectorAll("[data-tab-panel]").forEach(function (panel) {
          panel.hidden = panel.getAttribute("data-tab-panel") !== target;
        });
      });
    });
  });

  /* ---------- Zone polygon editor ---------- */
  window.CampyZoneEditor = function (root, options) {
    options = options || {};
    // Prefer a dedicated editor SVG so a reference overlay in the same
    // container is never mistaken for the editing surface.
    var svg = root.querySelector(options.svg || "#zone-editor-svg") || root.querySelector("svg");
    var input = document.querySelector(options.input || "#id_polygon");
    var points = [];
    try { points = JSON.parse(input && input.value ? input.value : "[]") || []; } catch (e) { points = []; }

    function render() {
      var path = points.map(function (p) { return (p[0] * 100) + "," + (p[1] * 100); }).join(" ");
      var markup = "";
      if (points.length > 1) {
        markup += '<polygon class="zone-poly ' + (options.kind || "") + '" points="' + path + '" />';
      }
      points.forEach(function (p, index) {
        markup += '<circle class="zone-point" cx="' + (p[0] * 100) + '" cy="' + (p[1] * 100) +
                  '" r="1.1" data-index="' + index + '" />';
      });
      svg.innerHTML = markup;
      if (input) input.value = JSON.stringify(points.map(function (p) {
        return [Math.round(p[0] * 10000) / 10000, Math.round(p[1] * 10000) / 10000];
      }));
      var counter = document.querySelector(options.counter || "#zone-point-count");
      if (counter) counter.textContent = points.length;
    }

    root.addEventListener("click", function (event) {
      if (event.target.classList.contains("zone-point")) {
        points.splice(parseInt(event.target.getAttribute("data-index"), 10), 1);
        render();
        return;
      }
      var rect = root.getBoundingClientRect();
      points.push([
        Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
        Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1)
      ]);
      render();
    });

    var clear = document.querySelector(options.clear || "#zone-clear");
    if (clear) clear.addEventListener("click", function (event) {
      event.preventDefault(); points = []; render();
    });
    var undo = document.querySelector(options.undo || "#zone-undo");
    if (undo) undo.addEventListener("click", function (event) {
      event.preventDefault(); points.pop(); render();
    });

    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    render();
    return { points: function () { return points; }, render: render };
  };

  /* ---------- Auto-dismiss flash messages ---------- */
  document.querySelectorAll("[data-autodismiss]").forEach(function (element) {
    setTimeout(function () {
      element.style.transition = "opacity .4s";
      element.style.opacity = "0";
      setTimeout(function () { element.remove(); }, 400);
    }, parseInt(element.getAttribute("data-autodismiss"), 10) || 6000);
  });
})();
