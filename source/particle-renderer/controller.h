//
// Created by Cubolink on 27-05-2023.
//

#ifndef NBODIES_PROBLEM_CONTROLLER_H
#define NBODIES_PROBLEM_CONTROLLER_H

#include "camera.h"

class Controller {
private:
    Camera *camera;
    int buttonState;
    // float scaleFactor;
    int sw, sh;

public:
    explicit Controller(float scaleFactor, float sw, float sh)
    :buttonState(0), // scaleFactor(scaleFactor),
    sw(sw), sh(sh)
    {
        camera = new Camera(scaleFactor);
    }

    void updateCameraProperties()
    {
        camera->updateLag();
    }
    void setCameraOxOy(int ox, int oy)
    {
        camera->updateOxOy(ox, oy);
    }

    void cameraMotion(int x, int y)
    {
        if (buttonState == 3)
        {
            camera->zoomMotion(x, y);
        }
        else if (buttonState & 2)
        {
            camera->translateMotion(x, y, sw, sh);
        }
        else if (buttonState & 1)
        {
            camera->rotateMotion(x, y);
        }
    }

    void cameraReset(float scaleFactor)
    {
        camera->reset(scaleFactor);
    }

    void setButtonState(int newState)
    {
        buttonState = newState;
        camera->printPosition();
    }

    void setScreenSize(int newSw, int newSh)
    {
        sw = newSw;
        sh = newSh;
    }

    float getScreenWidth() const
    {
        return sw;
    }

    float getScreenHeight() const
    {
        return sh;
    }

    float* getCameraTransLag()
    {
        return camera->getTransLag();
    }

    float* getCameraRotLag()
    {
        return camera->getRotLag();
    }
};


#endif //NBODIES_PROBLEM_CONTROLLER_H
