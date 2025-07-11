<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watchEffect } from "vue";

import ChartDotsIcon from "@knime/styles/img/icons/chart-dots.svg";
import {
  type ExtensionConfig,
  UIExtension,
  type UIExtensionAPILayer,
} from "@knime/ui-extension-renderer/vue";
import {
  AlertingService,
  JsonDataService,
  type UIExtensionService,
} from "@knime/ui-extension-service";

import type { ViewResponse } from "@/types";

import MessageBox from "./chat/MessageBox.vue";

const props = defineProps<ViewResponse>();

const dataAvailable = ref<boolean>(false);

const renderError = ref<string | null>(null);

const extensionConfig = ref<ExtensionConfig | null>(null);
const resourceLocation = ref<string | null>(null);
const baseService = ref<UIExtensionService<UIExtensionAPILayer> | null>(null);

const parsedNodeID = computed(() => {
  const ids = props.content.split("#");
  if (ids[0] === "not-a-virtual-project") {
    return {
      error: `View for node ${ids[1]} can't be rendered. Debug mode enabled?`,
    };
  }
  const nodeId = ids[1];
  return {
    result: {
      projectId: ids[0],
      workflowId: nodeId.substring(0, nodeId.lastIndexOf(":")),
      nodeId,
    },
  };
});

const noop = () => {
  /* mock unused api fields */
};

const apiLayer: UIExtensionAPILayer = {
  registerPushEventService: () => () => {},
  callNodeDataService: async ({ dataServiceRequest, serviceType }) => {
    if (!parsedNodeID.value.result) {
      return { result: null };
    }
    const { projectId, workflowId, nodeId } = parsedNodeID.value.result;
    // TODO AP-24611: only works if the modern UI is the ui-extension embedder
    // but NOT in case of the pagebuilder
    const response = await baseService.value?.callKnimeUiApi!(
      "NodeService.callNodeDataService",
      {
        projectId,
        workflowId,
        versionId: "current-state",
        nodeId,
        extensionType: "view",
        serviceType,
        dataServiceRequest,
      },
    );

    return response?.isSome ? { result: response.result } : { result: null };
  },
  updateDataPointSelection: async () => {},
  getResourceLocation(path) {
    return Promise.resolve(
      (extensionConfig.value!.resourceInfo as unknown as { baseUrl: string })
        .baseUrl + path,
    );
  },
  imageGenerated: noop,
  onApplied: noop,
  onDirtyStateChange: noop,
  publishData: noop,
  sendAlert(alert) {
    AlertingService.getInstance().then((service) =>
      // @ts-expect-error please provide a comment
      service.baseService.sendAlert(alert),
    );
  },
  setControlsVisibility: noop,
  setReportingContent: noop,
  showDataValueView: noop,
  closeDataValueView: noop,
};

onMounted(async () => {
  baseService.value = (
    (await JsonDataService.getInstance()) as any
  ).baseService;
});

watchEffect(() => {
  if (baseService.value === null) {
    return;
  }

  if (!parsedNodeID.value.result) {
    dataAvailable.value = false;
    renderError.value = parsedNodeID.value.error ?? "View can't be rendered";
    return;
  }
  const { projectId, workflowId, nodeId } = parsedNodeID.value.result;
  // TODO AP-24611: only works if the modern UI is the ui-extension embedder
  // but NOT in case of the pagebuilder
  baseService.value.callKnimeUiApi!("NodeService.getNodeView", {
    projectId,
    workflowId,
    versionId: "current-state",
    nodeId,
  })
    .then((response) => {
      if (response.isSome) {
        extensionConfig.value = response.result;
        resourceLocation.value =
          // TODO AP-24611: baseUrl is not provided in case of browser editing
          // and data-apps
          // @ts-expect-error please provide a comment
          extensionConfig.value?.resourceInfo?.baseUrl +
          extensionConfig.value?.resourceInfo?.path;
        dataAvailable.value = true;
      } else {
        renderError.value = "View can't be rendered";
        dataAvailable.value = false;
      }
    })
    .catch((error) => {
      consola.error("Error while fetching data", error);
      dataAvailable.value = false;
    });
});

onUnmounted(() => {
  if (parsedNodeID.value.result) {
    const { projectId, workflowId, nodeId } = parsedNodeID.value.result;
    // TODO AP-24611: only works if the modern UI is the ui-extension embedder
    // but NOT in case of the pagebuilder
    baseService.value?.callKnimeUiApi!(
      "NodeService.deactivateNodeDataServices",
      {
        projectId,
        workflowId,
        versionId: "current-state",
        nodeId,
      },
    );
    dataAvailable.value = false;
  }
});
</script>

<template>
  <MessageBox class="message" is-node-view>
    <template #icon>
      <ChartDotsIcon />
    </template>
    <template #name> Node View </template>
    <UIExtension
      v-if="dataAvailable"
      :api-layer="apiLayer"
      :extension-config="extensionConfig!"
      :resource-location="resourceLocation!"
      :shadow-app-style="{ height: '100%', width: '100%', overflowX: 'scroll' }"
    />
    <div v-if="renderError">{{ renderError }}</div>
  </MessageBox>
</template>
