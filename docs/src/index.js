import { WebViewer } from "@rerun-io/web-viewer";

const rrdUrl = "../JOSH/recordings/test_demo_2.rrd";
const parentElement = document.getElementById("viewer-container");

const viewer = new WebViewer();
await viewer.start(rrdUrl, parentElement);
