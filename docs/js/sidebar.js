// The readthedocs theme brings the current page's sidebar entry into view with scrollIntoView(),
// which scrolls every scrollable ancestor, the page included, so the page jumps to line up with the
// entry. Keep the theme's highlighting and expanding, but scroll only the sidebar, and only when
// the entry is out of its view.
(function () {
  var nav = window.SphinxRtdTheme && window.SphinxRtdTheme.Navigation;
  if (!nav || typeof nav.reset !== "function") {
    return;
  }
  var reset = nav.reset;
  nav.reset = function () {
    var scrollIntoView = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function () {
      var sidebar = document.querySelector("div.wy-side-scroll");
      if (!sidebar || !sidebar.contains(this)) {
        return scrollIntoView.apply(this, arguments);
      }
      var top = this.getBoundingClientRect().top - sidebar.getBoundingClientRect().top;
      if (top < 0 || top > sidebar.clientHeight - this.offsetHeight) {
        sidebar.scrollTop += top - sidebar.clientHeight / 3;
      }
    };
    try {
      return reset.apply(this, arguments);
    } finally {
      Element.prototype.scrollIntoView = scrollIntoView;
    }
  };
})();
