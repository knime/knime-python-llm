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

interface Props {
  viewNodeIds: string[];
}

const props = withDefaults(defineProps<Props>(), {});

const dataAvailable = ref<boolean>(false);

const extensionConfig = ref<ExtensionConfig | null>(null);
const resourceLocation = ref<string | null>(null);
const baseService = ref<UIExtensionService<UIExtensionAPILayer> | null>(null);

const noop = () => {
  /* mock unused api fields */
};
const replaceRootIndex = (nodeId: string): string => {
  return nodeId.replace(/^\d+:/, "root:");
};
const apiLayer: UIExtensionAPILayer = {
  registerPushEventService: () => () => {},
  callNodeDataService: async ({ dataServiceRequest, serviceType }) => {
    const response = await baseService.value?.callKnimeUiApi!(
      "NodeService.callNodeDataService",
      {
        nodeId: replaceRootIndex(props.viewNodeIds[0]),
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

  baseService.value.callKnimeUiApi!("NodeService.getNodeView", {
    nodeId: replaceRootIndex(props.viewNodeIds[0]),
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
  baseService.value?.callKnimeUiApi!("NodeService.deactivateNodeDataServices", {
    nodeId: replaceRootIndex(props.viewNodeIds[0]),
  });
  dataAvailable.value = false;
});
</script>

<template>
  <div class="port-table">
    <UIExtension
      v-if="dataAvailable"
      :api-layer="apiLayer"
      :extension-config="extensionConfig!"
      :resource-location="resourceLocation!"
      :shadow-app-style="{ height: '100%', width: '100%', overflowX: 'scroll' }"
    />
  </div>
</template>

<style lang="postcss" scoped>
.port-table {
  height: 100%;
  width: 100%;
}
</style>
