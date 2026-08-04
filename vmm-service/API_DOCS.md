# VMM图像音乐生成API文档

本文档描述了Visual Music Model (VMM) 的API接口，用于从图像生成音乐，以及从图像结合文字描述生成音乐。

## API端点

VMM提供了两个API端点：

1. `/api/generate_with_image` - 仅使用图像生成音乐
2. `/api/generate_with_image_and_text` - 使用图像和文字描述一起生成音乐

## 1. 仅图像生成音乐

### 请求

- **URL**: `/api/generate_with_image`
- **方法**: POST
- **内容类型**: multipart/form-data

### 参数

| 参数名   | 类型    | 必需 | 描述                              |
|----------|---------|------|-----------------------------------|
| image    | 文件    | 是   | 要处理的图像文件（JPG、PNG等格式）|
| duration | 数字    | 否   | 生成音乐的时长（秒），默认为30秒  |

### 示例请求

```bash
curl -X POST \
  http://localhost:5000/api/generate_with_image \
  -F "image=@path/to/your/image.jpg" \
  -F "duration=30"
```

### 响应

```json
{
  "success": true,
  "audio_url": "/audios/generated_music_1234567890.wav",
  "timestamp": 1234567890
}
```

## 2. 图像和文字描述生成音乐

### 请求

- **URL**: `/api/generate_with_image_and_text`
- **方法**: POST
- **内容类型**: multipart/form-data

### 参数

| 参数名          | 类型    | 必需 | 描述                              |
|-----------------|---------|------|-----------------------------------|
| image           | 文件    | 是   | 要处理的图像文件（JPG、PNG等格式）|
| text_description| 字符串  | 是   | 音乐的文字描述，用于增强生成效果  |
| duration        | 数字    | 否   | 生成音乐的时长（秒），默认为30秒  |

### 文字描述要求

`text_description` 参数应该是一个描述所需音乐风格、情绪、节奏或内容的文字描述。建议：

- 描述具体的音乐风格（例如："轻柔的钢琴曲"，"激烈的摇滚音乐"）
- 描述音乐情绪（例如："欢快的"，"悲伤的"，"神秘的"）
- 描述所需的乐器（例如："小提琴独奏"，"管弦乐团"）
- 描述与图像相关的场景或故事（例如："海边日落的轻松氛围"）
- 长度控制在200个字符以内，简洁明了

### 示例请求

```bash
curl -X POST \
  http://localhost:5000/api/generate_with_image_and_text \
  -F "image=@path/to/your/image.jpg" \
  -F "text_description=悠扬的钢琴曲，伴随着轻柔的弦乐，营造出宁静而温馨的氛围" \
  -F "duration=30"
```

### 响应

```json
{
  "success": true,
  "audio_url": "/audios/generated_music_1234567890.wav",
  "timestamp": 1234567890,
  "text_description": "悠扬的钢琴曲，伴随着轻柔的弦乐，营造出宁静而温馨的氛围"
}
```

## 错误响应

当请求出现问题时，API将返回一个错误响应：

```json
{
  "success": false,
  "message": "错误描述"
}
```

常见错误包括：

- "缺少图像文件" - 请求中没有包含图像文件
- "缺少文字描述" - 在使用图像和文字描述生成音乐时没有提供文字描述
- "音乐生成失败: ..." - 在处理过程中发生了错误

## 注意事项

1. 音频文件存储在服务器的公共文件夹中，可以通过返回的audio_url访问
2. 生成过程可能需要较长时间，特别是在生成较长音频时
3. 图像应该清晰可辨，建议使用分辨率大于500x500像素的图像
4. 文字描述应该尽量具体且描述性强，这将有助于生成符合预期的音乐 