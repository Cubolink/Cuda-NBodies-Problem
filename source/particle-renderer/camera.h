//
// Created by Cubolink on 27-05-2023.
//

#ifndef NBODIES_PROBLEM_CAMERA_H
#define NBODIES_PROBLEM_CAMERA_H

#include <cmath>

class Camera
{
private:
    int ox, oy;
    const float inertia;
    float 	camera_trans[3];
    float 	camera_trans_lag[3];
    float 	camera_rot[3];
    float 	camera_rot_lag[3];
public:
    explicit Camera(float scaleFactor)
    : ox(0), oy(0), inertia(0.1),
    camera_trans{0, 6*scaleFactor, -45*scaleFactor},
    camera_trans_lag{0, 6*scaleFactor, -45*scaleFactor},
    camera_rot{0, 0, 0},
    camera_rot_lag{0, 0, 0}
    {

    }
    void updateLag() {
        for (int i = 0; i < 3; i++)
        {
            camera_trans_lag[i] += (camera_trans[i] - camera_trans_lag[i]) * inertia;
            camera_rot_lag[i] += (camera_rot[i] - camera_rot_lag[i]) * inertia;
        }
    }
    void updateOxOy(int newOx, int newOy) {
        ox = newOx;
        oy = newOy;
    }
    void zoomMotion(int x, int y) {
        auto dy = (float) (y - oy);

        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);

        ox = x;
        oy = y;
    }
    void translateMotion(int x, int y, int sw, int sh) {
        auto dx = (float) (x - ox);
        auto dy = (float) (y - oy);

        camera_trans[0] += 0.005f * fabs(camera_trans[2]) * dx * (720.0f/(float) sw) / 2.0f;
        camera_trans[1] -= 0.005f * fabs(camera_trans[2]) * dy * (480.0f/(float) sh) / 2.0f;

        ox = x;
        oy = y;
    }
    void rotateMotion(int x, int y) {
        auto dx = (float) (x - ox);
        auto dy = (float) (y - oy);

        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;

        ox = x;
        oy = y;
    }
    void reset(float scaleFactor) {
        camera_trans[0] = 0;
        camera_trans[1] = 6 * scaleFactor;
        camera_trans[2] = -45 * scaleFactor;

        camera_trans_lag[0] = 0;
        camera_trans_lag[1] = 6 * scaleFactor;
        camera_trans_lag[2] = -45 * scaleFactor;

        camera_rot[0] = camera_rot[1] = camera_rot[2] = 0;

        camera_rot_lag[0] = camera_rot_lag[1] = camera_rot_lag[2] = 0;
    }
    float* getTransLag() {
        return camera_trans_lag;
    }
    float* getRotLag() {
        return camera_rot_lag;
    }

};


#endif //NBODIES_PROBLEM_CAMERA_H
