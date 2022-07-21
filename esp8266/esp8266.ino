#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

#define IN1 5 
#define IN2 4 
#define ENA 14

#define WIFI_SSID "pi"
#define WIFI_PASS "12345678"
#define UDP_PORT 4444
 
WiFiUDP UDP;
char packet[255];
  
void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  
  Serial.begin(115200);
  Serial.println();
  
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  
  Serial.print("Connecting to ");
  Serial.print(WIFI_SSID);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(100);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
 
  UDP.begin(UDP_PORT);
}
 
void loop() {
  int packetSize = UDP.parsePacket();
  if (packetSize) {
    int len = UDP.read(packet, 255);
    if (len > 0) packet[len] = '\0';
    
    switch (packet[0]){
      case 'M':
        int SPD = 0;
        SPD = SPD + (packet[4] - '0') * 100;
        SPD = SPD + (packet[5] - '0') * 10;
        SPD = SPD + (packet[6] - '0');
        if (packet[2] == 'F') goForward(SPD);
        if (packet[2] == 'B') goBackward(SPD);
    }
  }
}

void goForward(int SPD){
  analogWrite(ENA, SPD);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW); 
}

void goBackward(int SPD){
  analogWrite(ENA, SPD);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
}
