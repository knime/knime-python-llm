import { createApp } from "vue";

import "./style.css";
import App from "./App.vue";
import { setupLogger } from "./plugins/logger";

// Setup logger for production
setupLogger();

createApp(App).mount("#app");
