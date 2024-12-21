import { WebViewer } from "@rerun-io/web-viewer";

const rrdUrl = "https://raw.githubusercontent.com/genforce/JOSH/main/recordings/demo_1.rrd";
const parentElement = document.getElementById("demo1");

const viewer = new WebViewer();
window.addEventListener('DOMContentLoaded', function () {
    // Remove all autofocus attributes
    const autofocusElements = document.querySelectorAll('[autofocus]');
    autofocusElements.forEach(element => {
      element.removeAttribute('autofocus');
    });
  
    // Prevent scrolling to focused elements
    const originalFocus = HTMLElement.prototype.focus;
    HTMLElement.prototype.focus = function (options) {
      // Allow focus for debugging or specific cases if needed
      if (!this.classList.contains('allow-focus')) {
        return; // Block focus call
      }
      originalFocus.call(this, options);
    };
  });
await viewer.start(rrdUrl, parentElement, {render_backend: "webgl", hide_welcome_screen: true, panel_state_overrides: {top: "Hidden", blueprint: "Hidden"}});
