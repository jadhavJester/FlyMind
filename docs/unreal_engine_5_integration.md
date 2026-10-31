# FlyMind: Unreal Engine 5 Real-Time Camera Streamer & Integration Guide

This guide details how to build and execute the **Minimal Working Experiment** that streams real-time simulated camera frames from **Unreal Engine 5** directly into FlyMind's Python biological intelligence backend.

---

## 1. Architectural Concept

```
┌──────────────────────────────────────────────┐
│            UNREAL ENGINE 5 VIEWPORT          │
│                                              │
│  [Drone Pawn]                                │
│       └── [SceneCaptureComponent2D]          │
│                 │ (Renders 160x120 RGB)      │
│                 ▼                            │
│         [TextureRenderTarget2D]              │
│                 │                            │
│                 ▼                            │
│    [FlyMindCameraStreamer (C++ Actor)]       │
│                 │ (Extracts Raw Color Bytes) │
│                 │ (Prepend 16-byte Header)   │
└─────────────────┼────────────────────────────┘
                  │  TCP Socket Stream (127.0.0.1:9876)
                  ▼
┌──────────────────────────────────────────────┐
│         FLYMIND PYTHON PERCEPTION NODE       │
│      (flymind/simulation/unreal/ue5_server)  │
│                                              │
│  1. Unpacks Frame Header [FLYM, W, H, Time]  │
│  2. Farnebäck Dense Optical Flow             │
│  3. 2D Hassenstein-Reichardt EMDs (T4/T5)    │
│  4. LPLC2 4-Lobed Cross Convolution          │
│  5. 2D Flow Divergence Relative Nearness     │
│  6. Spiking Giant Fiber LIF Looming Trigger  │
│                                              │
│  Displays 4-Quadrant Live Research HUD:      │
│  [Camera Stream]   [Nearness Proximity Map]  │
│  [T4/T5 Channels]  [Giant Fiber Looming HUD] │
└──────────────────────────────────────────────┘
```

---

## 2. Wire Protocol Specification (16-Byte Header)

Every packet sent from Unreal Engine 5 to FlyMind consists of:
1. **Header (16 bytes, Big-Endian `!4sIII`)**:
   * `Magic`: 4 bytes ASCII (`b"FLYM"`)
   * `Width`: 4 bytes `uint32` (e.g. `160`)
   * `Height`: 4 bytes `uint32` (e.g. `120`)
   * `Timestamp_ms`: 4 bytes `uint32` (Simulation time in milliseconds)
2. **Payload**:
   * Exactly `Width * Height * 3` bytes of raw contiguous RGB/BGR image data.

---

## 3. Unreal Engine 5 C++ Implementation

### Step 1: Add Networking to Your UE5 Project Build File
In your project's `MyProject.Build.cs`, add `"Sockets"` and `"Networking"` to `PublicDependencyModuleNames`:
```csharp
PublicDependencyModuleNames.AddRange(new string[] {
    "Core", "CoreUObject", "Engine", "InputCore", "Sockets", "Networking", "RenderCore"
});
```

### Step 2: Create `FlyMindCameraStreamer.h`
Add a new C++ Actor Component to your Drone Pawn in Unreal Engine 5:

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "FlyMindCameraStreamer.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class FLYMIND_API UFlyMindCameraStreamer : public UActorComponent
{
    GENERATED_BODY()

public:
    UFlyMindCameraStreamer();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FlyMind")
    UTextureRenderTarget2D* RenderTarget;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FlyMind")
    FString ServerIP = "127.0.0.1";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FlyMind")
    int32 ServerPort = 9876;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FlyMind")
    float TargetFPS = 50.0f;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    FSocket* ClientSocket;
    FTimerHandle StreamTimerHandle;
    TArray<FColor> ColorBuffer;
    TArray<uint8> SendBuffer;

    void ConnectToServer();
    void StreamFrame();
};
```

### Step 3: Implement `FlyMindCameraStreamer.cpp`

```cpp
#include "FlyMindCameraStreamer.h"
#include "Async/Async.h"

UFlyMindCameraStreamer::UFlyMindCameraStreamer()
{
    PrimaryComponentTick.bCanEverTick = false;
    ClientSocket = nullptr;
}

void UFlyMindCameraStreamer::BeginPlay()
{
    Super::BeginPlay();
    ConnectToServer();

    if (ClientSocket && RenderTarget)
    {
        float Interval = 1.0f / FMath::Max(1.0f, TargetFPS);
        GetWorld()->GetTimerManager().SetTimer(StreamTimerHandle, this, &UFlyMindCameraStreamer::StreamFrame, Interval, true);
    }
}

void UFlyMindCameraStreamer::ConnectToServer()
{
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem) return;

    TSharedRef<FInternetAddr> Addr = SocketSubsystem->CreateInternetAddr();
    bool bIsValid;
    Addr->SetIp(*ServerIP, bIsValid);
    Addr->SetPort(ServerPort);

    if (bIsValid)
    {
        ClientSocket = SocketSubsystem->CreateSocket(NAME_Stream, TEXT("FlyMindSocket"), false);
        if (ClientSocket && ClientSocket->Connect(*Addr))
        {
            UE_LOG(LogTemp, Log, TEXT("[FlyMind] Connected to Python Bridge on %s:%d"), *ServerIP, ServerPort);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("[FlyMind] Failed to connect to Python Bridge on %s:%d"), *ServerIP, ServerPort);
        }
    }
}

void UFlyMindCameraStreamer::StreamFrame()
{
    if (!ClientSocket || !RenderTarget) return;

    FRenderTarget* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
    if (!RenderTargetResource) return;

    int32 Width = RenderTarget->SizeX;
    int32 Height = RenderTarget->SizeY;

    if (!RenderTargetResource->ReadPixels(ColorBuffer)) return;

    // Header: Magic 'FLYM', Width, Height, Timestamp_ms
    uint32 Timestamp = (uint32)(GetWorld()->GetTimeSeconds() * 1000.0f);
    SendBuffer.SetNumUninitialized(16 + (Width * Height * 3));

    // Big-endian magic
    SendBuffer[0] = 'F'; SendBuffer[1] = 'L'; SendBuffer[2] = 'Y'; SendBuffer[3] = 'M';
    FMemory::Memcpy(&SendBuffer[4], &Width, 4);
    FMemory::Memcpy(&SendBuffer[8], &Height, 4);
    FMemory::Memcpy(&SendBuffer[12], &Timestamp, 4);

    // Pack BGR pixels
    uint8* PixelDest = &SendBuffer[16];
    for (int32 i = 0; i < ColorBuffer.Num(); ++i)
    {
        PixelDest[i * 3 + 0] = ColorBuffer[i].B;
        PixelDest[i * 3 + 1] = ColorBuffer[i].G;
        PixelDest[i * 3 + 2] = ColorBuffer[i].R;
    }

    int32 BytesSent = 0;
    ClientSocket->Send(SendBuffer.GetData(), SendBuffer.Num(), BytesSent);
}

void UFlyMindCameraStreamer::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    if (ClientSocket)
    {
        ClientSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ClientSocket);
        ClientSocket = nullptr;
    }
}
```

---

## 4. Running the Minimal Working Experiment

You can immediately test and verify the entire end-to-end communication and visual intelligence pipeline without needing Unreal Engine installed yet, using FlyMind's built-in mock streamer:

### Step 1: Start the Python Bridge Server
In a terminal, run:
```powershell
python -m flymind.simulation.unreal.ue5_bridge_server
```
You will see:
```text
=======================================================
 FlyMind UE5 Bridge Server listening on 127.0.0.1:9876
 Waiting for Unreal Engine 5 client connection...
=======================================================
```

### Step 2: Start the Camera Stream Client
In a second terminal, run:
```powershell
python -m flymind.simulation.unreal.ue5_mock_client
```
The server will connect, ingest real-time 3D camera frames at $50\text{ Hz}$, process them through the biological pipeline (HR-EMD $\to$ LPLC2 $\to$ Giant Fiber $\to$ Flow Divergence), and display the live **4-Quadrant Research HUD** window!
