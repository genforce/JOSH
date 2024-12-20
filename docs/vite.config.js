import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

/** @type {import("vite").UserConfig} */
const config = {
  plugins: [wasm(), topLevelAwait()],
  // https://github.com/rerun-io/rerun/issues/6815
  optimizeDeps: {
    exclude: process.env.NODE_ENV === "production" ? [] : ["@rerun-io/web-viewer"],
  },
  server: {
    port: 5173,
  },
};


 
config.base = `/JOSH/`;
 

export default config;

