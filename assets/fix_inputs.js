// Kill white backgrounds on any input + force readable colors on dropdown
// menus that React-Select portals to <body>. CSS !important is sometimes
// beaten by Emotion-injected inline styles, so we patch the live DOM.
(function() {
    function isWhite(bg) {
        return bg === 'white' || bg === '#ffffff' || bg === '#fff' ||
               bg === 'rgb(255, 255, 255)' || bg === 'rgb(255,255,255)';
    }

    function fixInputs() {
        // Inputs everywhere
        document.querySelectorAll('input').forEach(function(el) {
            var bg = el.style.backgroundColor || window.getComputedStyle(el).backgroundColor;
            if (isWhite(bg)) {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
                el.style.setProperty('color', '#e0e0e0', 'important');
            }
            if (el.closest('.dash-dropdown') || el.closest('.Select') || el.closest('[class*="control"]')) {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
                el.style.setProperty('color', '#e0e0e0', 'important');
                el.style.setProperty('box-shadow', 'none', 'important');
                el.style.setProperty('outline', 'none', 'important');
            }
        });

        // Any divs with white bg inside in-page dropdowns
        document.querySelectorAll('.dash-dropdown div, .Select div').forEach(function(el) {
            var bg = window.getComputedStyle(el).backgroundColor;
            if (isWhite(bg)) {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
            }
        });

        // ── Dash 4 dropdown options (anchors, not divs) ──
        // The Dash 4 dropdown uses <a class="dash-dropdown-option"> inside
        // .dash-dropdown-options. Force readable colors on every option link.
        document.querySelectorAll('.dash-dropdown-options, .dash-dropdown-content').forEach(function(menu) {
            menu.style.setProperty('background-color', '#000000', 'important');
            menu.style.setProperty('background', '#000000', 'important');
            menu.style.setProperty('border', '1px solid #2d2d50', 'important');
            menu.querySelectorAll('a, .dash-dropdown-option, div, span, li, button').forEach(function(el) {
                var cls = el.className || '';
                var clsStr = (typeof cls === 'string') ? cls : '';
                var isFocused = clsStr.indexOf('focused') >= 0 ||
                                clsStr.indexOf('selected') >= 0 ||
                                el.getAttribute('aria-selected') === 'true';
                if (isFocused) {
                    el.style.setProperty('color', '#ff8800', 'important');
                    el.style.setProperty('background-color', '#1a1a2a', 'important');
                    el.style.setProperty('background', '#1a1a2a', 'important');
                } else {
                    el.style.setProperty('color', '#e0e0e0', 'important');
                    var bg = window.getComputedStyle(el).backgroundColor;
                    if (isWhite(bg)) {
                        el.style.setProperty('background-color', '#000000', 'important');
                        el.style.setProperty('background', '#000000', 'important');
                    }
                }
                el.style.setProperty('text-decoration', 'none', 'important');
            });
        });
    }

    // Run on load
    fixInputs();

    // Watch for DOM changes (dropdowns opening, React re-renders)
    var _debounceTimer;
    var observer = new MutationObserver(function() {
        clearTimeout(_debounceTimer);
        _debounceTimer = setTimeout(function() {
            try { fixInputs(); } catch(e) { /* silent */ }
        }, 50);
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
})();
