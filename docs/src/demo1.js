import { WebViewer } from "@rerun-io/web-viewer";

const rrdUrl = "https://raw.githubusercontent.com/genforce/JOSH/main/recordings/demo_1.rrd";
const parentElement = document.getElementById("demo1");

const viewer = new WebViewer();
let userScroll = false; // Flag to detect user scroll

// Monitor user scroll events
window.addEventListener('scroll', () => {
  userScroll = true; // Set flag when user scrolls
});

window.addEventListener('DOMContentLoaded', function () {
  // Prevent scrolling caused by focus during load
  const originalFocus = HTMLElement.prototype.focus;
  HTMLElement.prototype.focus = function (options) {
    if (!this.classList.contains('allow-focus')) {
      return; // Block focus calls during load
    }
    originalFocus.call(this, options);
  };

  // Ensure scroll behavior only happens if user hasn't scrolled
  window.addEventListener('load', () => {
    if (!userScroll) {
      window.scrollTo(0, 0); // Keep at top only if user hasn't scrolled
    }
  });
});
await viewer.start(rrdUrl, parentElement, {render_backend: "webgl", hide_welcome_screen: true, panel_state_overrides: {top: "Hidden", blueprint: "Hidden"}});
