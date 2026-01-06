#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "wifi_model.h"
#include <WiFi.h>
#include <WebServer.h>

// TinyML setup
tflite::MicroErrorReporter micro_error_reporter;
constexpr int kTensorArenaSize = 40 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

const char* label_names[] = {"Class_A", "Class_B", "Class_C"};

const char* ssid = "YOUR WIFI NAME";
const char* password = "PASSWORD";
WebServer server(80);

String html_ssid[4];
float html_rssi[4];
float html_normalized[4];
String html_strength[4];
String html_predicted_label;

void setup() {
  Serial.begin(115200);
  delay(2000);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());

  const tflite::Model* model = tflite::GetModel(wifi_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    &micro_error_reporter,
    nullptr,
    nullptr
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded");

  server.on("/", []() {
    String page = "<!DOCTYPE html><html><head><title>ESP32 Wi-Fi Dashboard</title>";
    page += "<meta http-equiv='refresh' content='5'>";
    page += "<style>body{font-family:Arial;} table{border-collapse:collapse;} td,th{border:1px solid #444;padding:8px;}</style>";
    page += "</head><body><h2>ESP32 Wi-Fi Scan & Model Prediction</h2>";
    page += "<table><tr><th>Network</th><th>SSID</th><th>RSSI</th><th>Normalized</th><th>Strength</th></tr>";
    for (int i = 0; i < 4; i++) {
      page += "<tr><td>" + String(i) + "</td><td>" + html_ssid[i] + "</td><td>" + String(html_rssi[i],2) + "</td>";
      page += "<td>" + String(html_normalized[i],3) + "</td><td>" + html_strength[i] + "</td></tr>";
    }
    page += "</table>";
    page += "<h3>Predicted label: " + html_predicted_label + "</h3>";
    page += "<p>Page refreshes every 5 seconds</p>";
    page += "</body></html>";
    server.send(200, "text/html", page);
  });

  server.begin();
  Serial.println("Web server started");
}

void loop() {
  server.handleClient();
  int n = WiFi.scanNetworks();
  if (n < 4) {
    Serial.println("Not enough networks found");
    delay(2000);
    return;
  }
  float rssi_vals[4];
  float normalized[4];
  String ssids[4];

  for (int i = 0; i < 4; i++) {
    rssi_vals[i] = WiFi.RSSI(i);
    normalized[i] = (rssi_vals[i] + 100.0f) / 100.0f;
    ssids[i] = WiFi.SSID(i);

    html_ssid[i] = ssids[i];
    html_rssi[i] = rssi_vals[i];
    html_normalized[i] = normalized[i];

    if (rssi_vals[i] > -50) html_strength[i] = "Strong";
    else if (rssi_vals[i] > -70) html_strength[i] = "Medium";
    else html_strength[i] = "Weak";
  }
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < 4; i++) input->data.f[i] = normalized[i];
  } else if (input->type == kTfLiteInt8) {
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    for (int i = 0; i < 4; i++) input->data.int8[i] = (int8_t)(normalized[i] / scale + zero_point);
  }

  interpreter->Invoke();
  int output_size = output->dims->data[output->dims->size - 1];
  int label = 0;
  float max_val = (output->type == kTfLiteFloat32) ? output->data.f[0] : output->data.int8[0];
  for (int i = 1; i < output_size; i++) {
    float val = (output->type == kTfLiteFloat32) ? output->data.f[i] : output->data.int8[i];
    if (val > max_val) {
      max_val = val;
      label = i;
    }
  }
  html_predicted_label = label_names[label];
  Serial.println("Wi-Fi Networks & Model Input");
  for (int i = 0; i < 4; i++) {
    Serial.print("Network "); Serial.print(i);
    Serial.print(": "); Serial.print(ssids[i]);
    Serial.print(", RSSI="); Serial.print(rssi_vals[i], 2);
    Serial.print(", normalized="); Serial.print(normalized[i], 3);
    Serial.print(", Strength="); Serial.print(html_strength[i]);
    if (i == 0) Serial.print("  <-- first input to model");
    Serial.println();
  }

  Serial.print("Raw output: ");
  for (int i = 0; i < output_size; i++) {
    if (output->type == kTfLiteFloat32) Serial.print(output->data.f[i], 6);
    else if (output->type == kTfLiteInt8) Serial.print(output->data.int8[i]);
    Serial.print(" ");
  }
  Serial.println();
  Serial.print("Predicted label: ");
  Serial.println(label_names[label]);
  Serial.println();

  delay(2000);
}
