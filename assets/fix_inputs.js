// Kill white backgrounds on any input element — React-Select applies inline styles
// that CSS !important cannot override. This MutationObserver catches them.
(function() {
    function fixInputs() {
        document.querySelectorAll('input').forEach(function(el) {
            var bg = el.style.backgroundColor || window.getComputedStyle(el).backgroundColor;
            if (bg && (bg === 'white' || bg === '#ffffff' || bg === 'rgb(255, 255, 255)' ||
                       bg === '#fff' || bg === 'rgb(255,255,255)')) {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
                el.style.setProperty('color', '#e0e0e0', 'important');
            }
            // Also force any input inside a dropdown
            if (el.closest('.dash-dropdown') || el.closest('.Select') || el.closest('[class*="control"]')) {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
                el.style.setProperty('color', '#e0e0e0', 'important');
                el.style.setProperty('box-shadow', 'none', 'important');
                el.style.setProperty('outline', 'none', 'important');
            }
        });
        // Also hit any div wrapper that might have white bg inside dropdowns
        document.querySelectorAll('.dash-dropdown div, .Select div').forEach(function(el) {
            var bg = window.getComputedStyle(el).backgroundColor;
            if (bg === 'rgb(255, 255, 255)' || bg === 'white') {
                el.style.setProperty('background-color', '#000000', 'important');
                el.style.setProperty('background', '#000000', 'important');
            }
        });
    }

    // Run on load
    fixInputs();

    // Watch for DOM changes (dropdowns opening, React re-renders)
    var observer = new MutationObserver(function() {
        fixInputs();
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
})();
