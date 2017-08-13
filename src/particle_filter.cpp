/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define YAW_RATE_MINIMAL 1e-6

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.;
    particles.push_back(particle);
    weights.push_back(1.);
  }
  is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (auto & particle : particles) {
    if (abs(yaw_rate) < YAW_RATE_MINIMAL) {
//      double new_theta=particle.theta+delta_t*yaw_rate;
      particle.x += velocity * (cos(particle.theta)) * delta_t;
      particle.y += velocity * (sin(particle.theta)) * delta_t;
//      particle.theta=new_theta;
    } else {
      double new_theta = particle.theta + delta_t*yaw_rate;
      particle.x += velocity / yaw_rate * (sin(new_theta) - sin(particle.theta));
      particle.y -= velocity / yaw_rate * (cos(new_theta) - cos(particle.theta));
      particle.theta = new_theta;
    }
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (auto & obs_landmark : observations) {
    double shortest_distance = std::numeric_limits<double>::max();
    int nearest_id = -1;
    for (int i = 0; i < predicted.size(); i++) {
      auto predict_landmark = predicted[i];
      double distance = dist(obs_landmark.x, obs_landmark.y, predict_landmark.x, predict_landmark.y);
//      cout<<"distance "<<distance<<"\n";
      if (distance < shortest_distance) {
        shortest_distance = distance;
        nearest_id = i;
      }//end if
    }//end for obs
//    cout<<"nearest "<<nearest_id<<"\n";
    obs_landmark.id = nearest_id;
  }//end for pred

}

template <class T>
T square(T in) {
  return in * in;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  weights.clear();
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = sqrt(1. / (2. * M_PI * std_landmark[0] * std_landmark[1]));
  for (auto & particle : particles) {
    //convert obs coord from car based to global based
    vector<LandmarkObs> obs_landmarks;
    for (auto & obs : observations) {
      LandmarkObs global_obs;
      global_obs.x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
      global_obs.y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
      obs_landmarks.push_back(global_obs);
    }
    vector<LandmarkObs> pred_landmarks;
    for (auto map_landmark : map_landmarks.landmark_list) {
      double distance = dist(map_landmark.x_f, map_landmark.y_f, particle.x, particle.y);
      if (distance > sensor_range)continue;
      LandmarkObs pred_landmark;
      pred_landmark.x = map_landmark.x_f;
      pred_landmark.y = map_landmark.y_f;
      pred_landmark.id = map_landmark.id_i;
      pred_landmarks.push_back(pred_landmark);
    }

    dataAssociation(pred_landmarks, obs_landmarks);

    //caculate weight
    double weight = 1;
    for (auto & obs_landmark : obs_landmarks) {
      auto match_pred_landmark = pred_landmarks[obs_landmark.id];
      weight *= gauss_norm * exp(
                  -(square<double>(obs_landmark.x - match_pred_landmark.x) / (2 * sig_x * sig_x) + square<double>(obs_landmark.y - match_pred_landmark.y) / (2 * sig_y * sig_y))
                );
//      cout<<"weight "<<square<double>(obs_landmark.x - match_pred_landmark.x)/(2 * sig_x*sig_x) + square<double>(obs_landmark.y - match_pred_landmark.y)/(2 * sig_y *sig_y)<<"\n";
    }
    particle.weight = weight;
    weights.push_back(weight);

    //debug info
//    for
//    auto size=obs_landmarks.size();
    std::vector<int> associations;
    vector<double> sense_x;
    std::vector<double> sense_y;
    for (auto global_obs : obs_landmarks) {
      associations.push_back(pred_landmarks[global_obs.id].id);
      sense_x.push_back(global_obs.x);
      sense_y.push_back(global_obs.y);
    }
    SetAssociations(particle, associations, sense_x, sense_y);
//    gps_error_x();
  }


}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<> dist_i(weights.begin(), weights.end());
  vector<Particle> new_particles{particles.size()};
  for (auto & new_particle : new_particles) {
    auto index = dist_i(gen);
    new_particle = particles[index];
//    cout<<"index "<<index<<"\n";
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

//  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
