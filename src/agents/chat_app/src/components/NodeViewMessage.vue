<script setup lang="ts">
import { onMounted, onUnmounted, ref, watchEffect } from "vue";

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

const extensionConfig = ref<ExtensionConfig | null>(null);
const resourceLocation = ref<string | null>(null);
const baseService = ref<UIExtensionService<UIExtensionAPILayer> | null>(null);

const noop = () => {
  /* mock unused api fields */
};

const join = (
  segments: string[],
  startIndex: number,
  endIndex: number,
): string => {
  const range = segments.slice(startIndex, endIndex);
  return range.join(":");
};

const parseNodeID = (
  nodeIdString: string,
): { projectId: string; workflowId: string; nodeId: string } => {
  const ids = nodeIdString.split(":");
  return {
    projectId: join(ids, 0, 2),
    workflowId: join(ids, 1, 3),
    nodeId: join(ids, 1, ids.length),
  };
};
const apiLayer: UIExtensionAPILayer = {
  registerPushEventService: () => () => {},
  callNodeDataService: async ({ dataServiceRequest, serviceType }) => {
    const { projectId, workflowId, nodeId } = parseNodeID(props.content);
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
      // @ts-expect-error
      service.baseService.sendAlert(alert),
    );
  },
  setControlsVisibility: noop,
  setReportingContent: noop,
  showDataValueView: noop,
  closeDataValueView: noop,
};

onMounted(async () => {
  console.log("Mounting NodeViewMessage");
  baseService.value = (
    (await JsonDataService.getInstance()) as any
  ).baseService;
});

watchEffect(() => {
  if (baseService.value === null) {
    return;
  }

  const { projectId, workflowId, nodeId } = parseNodeID(props.content);
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
          // @ts-ignore
          extensionConfig.value?.resourceInfo?.baseUrl +
          extensionConfig.value?.resourceInfo?.path;
      }
      dataAvailable.value = true;
    })
    .catch((error) => {
      console.error("Error while fetching data", error);
      dataAvailable.value = false;
    });
});

onUnmounted(() => {
  const { projectId, workflowId, nodeId } = parseNodeID(props.content);
  baseService.value?.callKnimeUiApi!("NodeService.deactivateNodeDataServices", {
    projectId,
    workflowId,
    versionId: "current-state",
    nodeId,
  });
  dataAvailable.value = false;
});
</script>

<template>
  <MessageBox class="message">
    <UIExtension
      v-if="dataAvailable"
      :api-layer="apiLayer"
      :extension-config="extensionConfig!"
      :resource-location="resourceLocation!"
      :shadow-app-style="{ height: '100%', width: '100%', overflowX: 'scroll' }"
    />
  </MessageBox>
</template>

<style lang="postcss" scoped>
.message {
  & :deep(.message-box) {
    min-width: 100%;
  }
}
</style>
