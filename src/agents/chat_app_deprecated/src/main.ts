import { createApp } from "vue";
import { createPinia } from "pinia";

import "./style.css";
import App from "./App.vue";
import { setupLogger } from "./plugins/logger";

// Setup logger for production
setupLogger();

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);
app.mount("#app");
