import { WebViewer } from "@rerun-io/web-viewer";

const rrdUrl = "https://raw.githubusercontent.com/genforce/JOSH-webpage/main/recordings/demo_1.rrd";
const parentElement = document.getElementById("viewer-container");

const viewer = new WebViewer();
await viewer.start(rrdUrl, parentElement, {render_backend: "webgl", panel_state_overrides: {top: "Hidden", blueprint: "Hidden"}});
