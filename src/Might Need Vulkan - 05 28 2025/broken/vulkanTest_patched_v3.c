#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define WIDTH 800
#define HEIGHT 600
#define VALIDATION_ENABLED 1

typedef struct {
    HINSTANCE hInstance;
    HWND hwnd;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkImage* swapChainImages;
    uint32_t swapChainImageCount;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkImageView* swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkFramebuffer* swapChainFramebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkSemaphore* imageAvailableSemaphores;
    VkSemaphore* renderFinishedSemaphores;
    VkFence* inFlightFences;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;
} VulkanApp;

typedef struct {
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    int graphicsFamilyHasValue;
    int presentFamilyHasValue;
} QueueFamilyIndices;

// Debug utilities
void printError(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "[ERROR] %s failed with code: %d\n", operation, result);
    }
}

void checkVkResult(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        printError(result, operation);
        exit(EXIT_FAILURE);
    }
}

void debugPrint(const char* message) {
    printf("[DEBUG] %s\n", message);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE:
            DestroyWindow(hwnd);
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

void initWindow(VulkanApp* app, const char* title) {
    app->hInstance = GetModuleHandle(NULL);

    WNDCLASSEX wc = {0};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = app->hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "VulkanWindowClass";
    RegisterClassEx(&wc);

    RECT rect = {0, 0, WIDTH, HEIGHT};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    app->hwnd = CreateWindowEx(
        0,
        "VulkanWindowClass",
        title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        NULL,
        NULL,
        app->hInstance,
        NULL
    );

    ShowWindow(app->hwnd, SW_SHOW);
    UpdateWindow(app->hwnd);
    debugPrint("Window initialized");
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = {0};
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);
    
    VkQueueFamilyProperties* queueFamilies = malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);
    
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
            indices.graphicsFamilyHasValue = 1;
        }
        
        VkBool32 presentSupport = 0;
        VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        checkVkResult(result, "vkGetPhysicalDeviceSurfaceSupportKHR");
        
        if (presentSupport) {
            indices.presentFamily = i;
            indices.presentFamilyHasValue = 1;
        }
        
        if (indices.graphicsFamilyHasValue && indices.presentFamilyHasValue) {
            break;
        }
    }
    
    free(queueFamilies);
    
    printf("[DEBUG] Found queue families: graphics=%d, present=%d\n",
           indices.graphicsFamily, indices.presentFamily);
    
    return indices;
}

void createInstance(VulkanApp* app) {
    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    const char* extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    };

    const char* validationLayers[] = {
        "VK_LAYER_KHRONOS_validation"
    };

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
    createInfo.ppEnabledExtensionNames = extensions;
    
#if VALIDATION_ENABLED
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers;
#else
    createInfo.enabledLayerCount = 0;
#endif

    VkResult result = vkCreateInstance(&createInfo, NULL, &app->instance);
    checkVkResult(result, "vkCreateInstance");
    debugPrint("Vulkan instance created");
}

void createSurface(VulkanApp* app) {
    VkWin32SurfaceCreateInfoKHR createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hinstance = app->hInstance;
    createInfo.hwnd = app->hwnd;

    VkResult result = vkCreateWin32SurfaceKHR(app->instance, &createInfo, NULL, &app->surface);
    checkVkResult(result, "vkCreateWin32SurfaceKHR");
    debugPrint("Window surface created");
}

void pickPhysicalDevice(VulkanApp* app) {
    uint32_t deviceCount = 0;
    VkResult result = vkEnumeratePhysicalDevices(app->instance, &deviceCount, NULL);
    checkVkResult(result, "vkEnumeratePhysicalDevices");
    
    if (deviceCount == 0) {
        fprintf(stderr, "Failed to find GPUs with Vulkan support!\n");
        exit(EXIT_FAILURE);
    }
    
    VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
    result = vkEnumeratePhysicalDevices(app->instance, &deviceCount, devices);
    checkVkResult(result, "vkEnumeratePhysicalDevices");
    
    for (uint32_t i = 0; i < deviceCount; i++) {
        QueueFamilyIndices indices = findQueueFamilies(devices[i], app->surface);
        if (indices.graphicsFamilyHasValue && indices.presentFamilyHasValue) {
            app->physicalDevice = devices[i];
            break;
        }
    }
    
    free(devices);
    
    if (app->physicalDevice == VK_NULL_HANDLE) {
        fprintf(stderr, "Failed to find a suitable GPU!\n");
        exit(EXIT_FAILURE);
    }
    
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(app->physicalDevice, &deviceProperties);
    printf("[DEBUG] Selected physical device: %s\n", deviceProperties.deviceName);
}

void createLogicalDevice(VulkanApp* app) {
    QueueFamilyIndices indices = findQueueFamilies(app->physicalDevice, app->surface);
    
    uint32_t uniqueQueueFamilies[2];
    uint32_t queueFamilyCount = 0;
    if (indices.graphicsFamily == indices.presentFamily) {
        uniqueQueueFamilies[0] = indices.graphicsFamily;
        queueFamilyCount = 1;
    } else {
        uniqueQueueFamilies[0] = indices.graphicsFamily;
        uniqueQueueFamilies[1] = indices.presentFamily;
        queueFamilyCount = 2;
    }
    
    VkDeviceQueueCreateInfo* queueCreateInfos = malloc(queueFamilyCount * sizeof(VkDeviceQueueCreateInfo));
    float queuePriority = 1.0f;
    
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        queueCreateInfos[i] = (VkDeviceQueueCreateInfo){0};
        queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[i].queueFamilyIndex = uniqueQueueFamilies[i];
        queueCreateInfos[i].queueCount = 1;
        queueCreateInfos[i].pQueuePriorities = &queuePriority;
    }
    
    VkPhysicalDeviceFeatures deviceFeatures = {0};
    
    const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    
    VkDeviceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = queueFamilyCount;
    createInfo.pQueueCreateInfos = queueCreateInfos;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = sizeof(deviceExtensions) / sizeof(deviceExtensions[0]);
    createInfo.ppEnabledExtensionNames = deviceExtensions;
    
#if VALIDATION_ENABLED
    const char* validationLayers[] = {
        "VK_LAYER_KHRONOS_validation"
    };
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers;
#else
    createInfo.enabledLayerCount = 0;
#endif
    
    printf("[DEBUG] Calling vkCreateDevice...\n");
    VkResult result = vkCreateDevice(app->physicalDevice, &createInfo, NULL, &app->device);
    checkVkResult(result, "vkCreateDevice");
    debugPrint("Logical device created");
    
    free(queueCreateInfos);
    
    vkGetDeviceQueue(app->device, indices.graphicsFamily, 0, &app->graphicsQueue);
    vkGetDeviceQueue(app->device, indices.presentFamily, 0, &app->presentQueue);
}

void createSwapChain(VulkanApp* app) {
    VkSurfaceCapabilitiesKHR capabilities;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app->physicalDevice, app->surface, &capabilities);
    checkVkResult(result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    
    uint32_t formatCount;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(app->physicalDevice, app->surface, &formatCount, NULL);
    checkVkResult(result, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    
    VkSurfaceFormatKHR* formats = malloc(formatCount * sizeof(VkSurfaceFormatKHR));
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(app->physicalDevice, app->surface, &formatCount, formats);
    checkVkResult(result, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    
    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (uint32_t i = 0; i < formatCount; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB && 
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat = formats[i];
            break;
        }
    }
    
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    
    VkExtent2D extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == UINT32_MAX) {
        extent.width = WIDTH;
        extent.height = HEIGHT;
    }
    
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }
    
    VkSwapchainCreateInfoKHR createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = app->surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    QueueFamilyIndices indices = findQueueFamilies(app->physicalDevice, app->surface);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily, indices.presentFamily};
    
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = NULL;
    }
    
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    
    result = vkCreateSwapchainKHR(app->device, &createInfo, NULL, &app->swapChain);
    checkVkResult(result, "vkCreateSwapchainKHR");
    debugPrint("Swapchain created");
    
    result = vkGetSwapchainImagesKHR(app->device, app->swapChain, &app->swapChainImageCount, NULL);
    checkVkResult(result, "vkGetSwapchainImagesKHR");
    app->swapChainImages = malloc(app->swapChainImageCount * sizeof(VkImage));
    result = vkGetSwapchainImagesKHR(app->device, app->swapChain, &app->swapChainImageCount, app->swapChainImages);
    checkVkResult(result, "vkGetSwapchainImagesKHR");
    
    app->swapChainImageFormat = surfaceFormat.format;
    app->swapChainExtent = extent;
    
    printf("[DEBUG] Swapchain created with %d images (%dx%d)\n", 
           app->swapChainImageCount, extent.width, extent.height);
    
    free(formats);
}

void createImageViews(VulkanApp* app) {
    app->swapChainImageViews = malloc(app->swapChainImageCount * sizeof(VkImageView));
    
    for (uint32_t i = 0; i < app->swapChainImageCount; i++) {
        VkImageViewCreateInfo createInfo = {0};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = app->swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = app->swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        
        VkResult result = vkCreateImageView(app->device, &createInfo, NULL, &app->swapChainImageViews[i]);
        checkVkResult(result, "vkCreateImageView");
    }
    debugPrint("Image views created");
}

void createRenderPass(VulkanApp* app) {
    VkAttachmentDescription colorAttachment = {0};
    colorAttachment.format = app->swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentReference colorAttachmentRef = {0};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    
    VkSubpassDependency dependency = {0};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    VkRenderPassCreateInfo renderPassInfo = {0};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    VkResult result = vkCreateRenderPass(app->device, &renderPassInfo, NULL, &app->renderPass);
    checkVkResult(result, "vkCreateRenderPass");
    debugPrint("Render pass created");
}

VkShaderModule createShaderModule(VulkanApp* app, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* code = malloc(fileSize);
    size_t readSize = fread(code, 1, fileSize, file);
    fclose(file);
    
    if (readSize != fileSize) {
        fprintf(stderr, "Failed to read shader file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = (uint32_t*)code;
    
    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(app->device, &createInfo, NULL, &shaderModule);
    checkVkResult(result, "vkCreateShaderModule");
    
    free(code);
    printf("[DEBUG] Shader module created from %s\n", filename);
    return shaderModule;
}

void createGraphicsPipeline(VulkanApp* app) {
    VkShaderModule vertShaderModule = createShaderModule(app, "shaders/vert.spv");
    VkShaderModule fragShaderModule = createShaderModule(app, "shaders/frag.spv");
    
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {0};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {0};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {0};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = NULL;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = NULL;
    
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {0};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    VkViewport viewport = {0};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)app->swapChainExtent.width;
    viewport.height = (float)app->swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    VkRect2D scissor = {0};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = app->swapChainExtent;
    
    VkPipelineViewportStateCreateInfo viewportState = {0};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    
    VkPipelineRasterizationStateCreateInfo rasterizer = {0};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    VkPipelineMultisampleStateCreateInfo multisampling = {0};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlending = {0};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = NULL;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = NULL;
    
    VkResult result = vkCreatePipelineLayout(app->device, &pipelineLayoutInfo, NULL, &app->pipelineLayout);
    checkVkResult(result, "vkCreatePipelineLayout");
    debugPrint("Pipeline layout created");
    
    VkGraphicsPipelineCreateInfo pipelineInfo = {0};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = app->pipelineLayout;
    pipelineInfo.renderPass = app->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    
    result = vkCreateGraphicsPipelines(app->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &app->graphicsPipeline);
    checkVkResult(result, "vkCreateGraphicsPipelines");
    debugPrint("Graphics pipeline created");
    
    vkDestroyShaderModule(app->device, fragShaderModule, NULL);
    vkDestroyShaderModule(app->device, vertShaderModule, NULL);
}

void createFramebuffers(VulkanApp* app) {
    app->swapChainFramebuffers = malloc(app->swapChainImageCount * sizeof(VkFramebuffer));
    
    for (uint32_t i = 0; i < app->swapChainImageCount; i++) {
        VkImageView attachments[] = {
            app->swapChainImageViews[i]
        };
        
        VkFramebufferCreateInfo framebufferInfo = {0};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = app->renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = app->swapChainExtent.width;
        framebufferInfo.height = app->swapChainExtent.height;
        framebufferInfo.layers = 1;
        
        VkResult result = vkCreateFramebuffer(app->device, &framebufferInfo, NULL, &app->swapChainFramebuffers[i]);
        checkVkResult(result, "vkCreateFramebuffer");
    }
    debugPrint("Framebuffers created");
}

void createCommandPool(VulkanApp* app) {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(app->physicalDevice, app->surface);
    
    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
    
    VkResult result = vkCreateCommandPool(app->device, &poolInfo, NULL, &app->commandPool);
    checkVkResult(result, "vkCreateCommandPool");
    debugPrint("Command pool created");
}

void createCommandBuffer(VulkanApp* app) {
    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = app->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    VkResult result = vkAllocateCommandBuffers(app->device, &allocInfo, &app->commandBuffer);
    checkVkResult(result, "vkAllocateCommandBuffers");
    debugPrint("Command buffer allocated");
}

void recordCommandBuffer(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = NULL;
    
    VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    checkVkResult(result, "vkBeginCommandBuffer");
    
    VkRenderPassBeginInfo renderPassInfo = {0};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = app->renderPass;
    renderPassInfo.framebuffer = app->swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset.x = 0;
    renderPassInfo.renderArea.offset.y = 0;
    renderPassInfo.renderArea.extent = app->swapChainExtent;
    
    VkClearValue clearColor = {{{0.2f, 0.2f, 0.2f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;
    
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->graphicsPipeline);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);
    
    result = vkEndCommandBuffer(commandBuffer);
    checkVkResult(result, "vkEndCommandBuffer");
    debugPrint("Command buffer recorded");
}


void createSyncObjects(VulkanApp* app) {
    app->imageAvailableSemaphores = malloc(app->swapChainImageCount * sizeof(VkSemaphore));
    app->renderFinishedSemaphores = malloc(app->swapChainImageCount * sizeof(VkSemaphore));
    app->inFlightFences = malloc(app->swapChainImageCount * sizeof(VkFence));

    VkSemaphoreCreateInfo semaphoreInfo = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };

    for (size_t i = 0; i < app->swapChainImageCount; i++) {
        checkVkResult(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->imageAvailableSemaphores[i]), "vkCreateSemaphore (imageAvailable)");
        checkVkResult(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->renderFinishedSemaphores[i]), "vkCreateSemaphore (renderFinished)");
        checkVkResult(vkCreateFence(app->device, &fenceInfo, NULL, &app->inFlightFences[i]), "vkCreateFence");
    }

    debugPrint("Sync objects created");
}


void drawFrame(VulkanApp* app) {
    uint32_t imageIndex;
    VkSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
    VkFence inFlightFence;

    VkResult result = vkAcquireNextImageKHR(app->device, app->swapChain, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE, &imageIndex);
    checkVkResult(result, "vkAcquireNextImageKHR");

    imageAvailableSemaphore = app->imageAvailableSemaphores[imageIndex];
    renderFinishedSemaphore = app->renderFinishedSemaphores[imageIndex];
    inFlightFence = app->inFlightFences[imageIndex];

    vkWaitForFences(app->device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(app->device, 1, &inFlightFence);

    vkResetCommandBuffer(app->commandBuffer, 0);
    recordCommandBuffer(app, app->commandBuffer, imageIndex);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &app->commandBuffer;

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    result = vkQueueSubmit(app->graphicsQueue, 1, &submitInfo, inFlightFence);
    checkVkResult(result, "vkQueueSubmit");

    VkPresentInfoKHR presentInfo = {0};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &app->swapChain;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(app->presentQueue, &presentInfo);
    checkVkResult(result, "vkQueuePresentKHR");
}

void cleanup(VulkanApp* app) {
    vkDestroySemaphore(app->device, app->renderFinishedSemaphore, NULL);
    vkDestroySemaphore(app->device, app->imageAvailableSemaphore, NULL);
    vkDestroyFence(app->device, app->inFlightFence, NULL);
    vkDestroyCommandPool(app->device, app->commandPool, NULL);
    
    for (uint32_t i = 0; i < app->swapChainImageCount; i++) {
        vkDestroyFramebuffer(app->device, app->swapChainFramebuffers[i], NULL);
        vkDestroyImageView(app->device, app->swapChainImageViews[i], NULL);
    }
    
    free(app->swapChainFramebuffers);
    free(app->swapChainImageViews);
    free(app->swapChainImages);
    
    vkDestroyPipeline(app->device, app->graphicsPipeline, NULL);
    vkDestroyPipelineLayout(app->device, app->pipelineLayout, NULL);
    vkDestroyRenderPass(app->device, app->renderPass, NULL);
    vkDestroySwapchainKHR(app->device, app->swapChain, NULL);
    vkDestroyDevice(app->device, NULL);
    vkDestroySurfaceKHR(app->instance, app->surface, NULL);
    vkDestroyInstance(app->instance, NULL);
}

void initVulkan(VulkanApp* app) {
    createInstance(app);
    createSurface(app);
    pickPhysicalDevice(app);
    createLogicalDevice(app);
    createSwapChain(app);
    createImageViews(app);
    createRenderPass(app);
    createGraphicsPipeline(app);
    createFramebuffers(app);
    createCommandPool(app);
    createCommandBuffer(app);
    createSyncObjects(app);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    VulkanApp app = {0};
    
    initWindow(&app, "Vulkan Gradient Triangle (Windows API)");
    initVulkan(&app);
    
    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
        
        drawFrame(&app);
    }
    
    vkDeviceWaitIdle(app.device);
    cleanup(&app);
    
    return (int)msg.wParam;
}