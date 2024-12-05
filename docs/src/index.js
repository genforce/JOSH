import { WebViewer } from "@rerun-io/web-viewer";

const rrdUrl = "https://app.rerun.io/version/0.20.3/examples/arkit_scenes.rrd";
const parentElement = document.getElementById("viewer-container");

const viewer = new WebViewer();
await viewer.start(rrdUrl, parentElement);
