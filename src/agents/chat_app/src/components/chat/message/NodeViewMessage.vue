<script setup lang="ts">
import { onMounted, onUnmounted, ref, watchEffect } from "vue";

import ChartDotsIcon from "@knime/styles/img/icons/chart-dots.svg";
import {
  type ExtensionConfig,
  UIExtension,
  type UIExtensionAPILayer,
} from "@knime/ui-extension-renderer/vue";
import {
  AlertingService,
  JsonDataService,
  ResourceService,
  type UIExtensionService,
} from "@knime/ui-extension-service";

import { useChatStore } from "@/stores/chat";
import type { ViewMessage, WorkflowInfo } from "@/types";

import MessageBox from "./MessageBox.vue";

const props = defineProps<ViewMessage>();

const dataAvailable = ref<boolean>(false);

const renderError = ref<string | null>(null);

const extensionConfig = ref<ExtensionConfig | null>(null);
const resourceLocation = ref<string | null>(null);
const baseService = ref<UIExtensionService<UIExtensionAPILayer> | null>(null);

const chatStore = useChatStore();

const resolvedNodeID = ref<{
  result?: WorkflowInfo & { nodeId: string };
}>({});

const noop = () => {
  /* mock unused api fields */
};

const apiLayer: UIExtensionAPILayer = {
  registerPushEventService: () => () => {},
  callNodeDataService: async ({ dataServiceRequest, serviceType }) => {
    if (!resolvedNodeID.value.result) {
      return { result: null };
    }
    const { projectId, workflowId, nodeId } = resolvedNodeID.value.result;
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
  async getResourceLocation(path) {
    return (await ResourceService.getInstance()).getResourceUrl(path);
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

  try {
    const { projectId, workflowId } =
      await chatStore.getCombinedToolsWorkflowInfo();
    const id = props.content;
    const nodeId = `${workflowId}:${id}`;
    resolvedNodeID.value = {
      result: {
        projectId,
        workflowId: nodeId.substring(0, nodeId.lastIndexOf(":")),
        nodeId,
      },
    };
  } catch (error) {
    consola.error("Error while fetching workflow info", error);
    renderError.value =
      "View can't be rendered. Error resolving node ID. Debug workflow deleted?";
    dataAvailable.value = false;
  }
});

watchEffect(() => {
  if (baseService.value === null) {
    return;
  }

  if (!resolvedNodeID.value.result) {
    return;
  }
  const { projectId, workflowId, nodeId } = resolvedNodeID.value.result;
  baseService.value.callKnimeUiApi!("NodeService.getNodeView", {
    projectId,
    workflowId,
    versionId: "current-state",
    nodeId,
  })
    .then(async (response) => {
      if (response.isSome && baseService.value) {
        extensionConfig.value = response.result;
        const path = extensionConfig.value?.resourceInfo?.path;
        if (path) {
          // NOTE: we can use the 'ResourceService' here because it's
          // a node-view in node-view nesting
          // (on desktop, different ui-extension types have different base-urls)
          resourceLocation.value = await (
            await ResourceService.getInstance()
          ).getResourceUrl(path);
          dataAvailable.value = true;
          return;
        }
      }
      renderError.value = "View can't be rendered";
      dataAvailable.value = false;
    })
    .catch((error) => {
      consola.error("Error while fetching data", error);
      dataAvailable.value = false;
    });
});

onUnmounted(() => {
  if (resolvedNodeID.value.result) {
    const { projectId, workflowId, nodeId } = resolvedNodeID.value.result;
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
    <template #name>View | {{ name }}</template>
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
