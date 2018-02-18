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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std_pos[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 1000;

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std_pos[0]);
  std::normal_distribution<double> dist_y(y, std_pos[1]);
  std::normal_distribution<double> dist_theta(theta, std_pos[2]);

  for (size_t i = 0; i < num_particles; i++) {
    Particle p;
    p.id     = i;
    p.x      = dist_x(gen);
    p.y      = dist_y(gen);
    p.theta  = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

//  printf("Initalized %d particles around pose (%f, %f, %f)\n", num_particles, x, y, theta / M_PI * 180);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  // http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (unsigned int i = 0; i < num_particles; i++) {
    Particle& p = particles[i];
    // printf("particle %d old pose: (x=%f, y=%f, theta=%f)\n", i, p.x, p.y, p.theta);
    // printf("velocity=%f, yaw_rate=%f\n", velocity, yaw_rate);
    float xf, yf, thetaf;
    if (std::abs(yaw_rate) > 1e-5) {
      xf = p.x + velocity / yaw_rate * (std::sin(p.theta + yaw_rate * delta_t) - std::sin(p.theta));
      yf = p.y + velocity / yaw_rate * (std::cos(p.theta) - std::cos(p.theta + yaw_rate * delta_t));
    } else {
      xf = p.x + velocity * std::cos(p.theta) * delta_t;
      yf = p.y + velocity * std::sin(p.theta) * delta_t;
    }
    thetaf = p.theta + yaw_rate * delta_t;

    p.x     = xf + dist_x(gen);
    p.y     = yf + dist_y(gen);
    p.theta = thetaf + dist_theta(gen);
    if (p.theta > 2 * M_PI) {
      p.theta -= 2 * M_PI;
    }
    if (p.theta < 0) {
      p.theta += 2 * M_PI;
    }
    // printf("particle %d new pose: (x=%f, y=%f, theta=%f)", i, p.x, p.y, p.theta);
  }
} // ParticleFilter::prediction

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  // observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  // implement this method and use it as a helper during the updateWeights phase.
  for (auto& obs : observations) {
    double min_dist = std::numeric_limits<double>::max();
    // Nearest neighbor search
    for (auto& p : predicted) {
      double d = dist(obs.x, obs.y, p.x, p.y);
      if (d < min_dist) {
        min_dist = d;
        obs.id   = p.id;
      }
    }
  }
}

void ParticleFilter::dataAssociation(const Map& map, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  // observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  // implement this method and use it as a helper during the updateWeights phase.
  for (auto& obs : observations) {
    double min_dist = std::numeric_limits<double>::max();
    // Nearest neighbor search
    for (auto& landmark : map.landmark_list) {
      double d = dist(obs.x, obs.y, landmark.x_f, landmark.y_f);
      if (d < min_dist) {
        min_dist = d;
        obs.id   = landmark.id_i;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  // more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  // according to the MAP'S coordinate system. You will need to transform between the two systems.
  // Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  // The following is a good resource for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement (look at equation
  // 3.33
  // http://planning.cs.uiuc.edu/node99.html
  double weights_sum = 0;
  for (Particle& particle : particles) {
    // This particle's location in map frame
    double xp     = particle.x;
    double yp     = particle.y;
    double thetap = particle.theta;

    // Transform the observation in car's frame into map frame
    std::vector<LandmarkObs> transformed_observations;
    for (const LandmarkObs& obs : observations) {
      // Observation in car's frame
      double xc = obs.x;
      double yc = obs.y;

      // Observation in map frame
      double xm = xc * std::cos(thetap) - yc * std::sin(thetap) + xp;
      double ym = xc * std::sin(thetap) + yc * std::cos(thetap) + yp;

      LandmarkObs transformed_obs;
      transformed_obs.id = -1;
      transformed_obs.x  = xm;
      transformed_obs.y  = ym;
      transformed_observations.push_back(transformed_obs);
    }

    // for (auto& transformed_obs : transformed_observations) {
    // transformed_obs.print();
    // }

    // Data association, assign landmark id to each transformed observation
    dataAssociation(map_landmarks, transformed_observations);
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (size_t i = 0; i < transformed_observations.size(); i++) {
      associations.push_back(transformed_observations[i].id);
      sense_x.push_back(transformed_observations[i].x);
      sense_y.push_back(transformed_observations[i].y);
    }
    setAssociations(particle, associations, sense_x, sense_y);

    // Calculate this particle's weight
    double log_prob      = 0;
    double gaussian_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    for (size_t i = 0; i < transformed_observations.size(); i++) {
      double dx       = particle.sense_x[i] - map_landmarks.landmark_list[particle.associations[i] - 1].x_f;
      double dy       = particle.sense_y[i] - map_landmarks.landmark_list[particle.associations[i] - 1].y_f;
      double exponent = dx * dx / (2 * std_landmark[0] * std_landmark[0])
                        + dy * dy / (2 * std_landmark[1] * std_landmark[1]);
      log_prob += std::log(gaussian_norm) - exponent;
      // printf("obs: id=%d, sense_x=%f, sense_y=%f\n", particle.associations[i], particle.sense_x[i],
      // particle.sense_y[i]);
      // printf("map: id=%d, x=%f, y=%f\n",
      // map_landmarks.landmark_list[particle.associations[i] - 1].id_i,
      // map_landmarks.landmark_list[particle.associations[i] - 1].x_f,
      // map_landmarks.landmark_list[particle.associations[i] - 1].y_f);
      // printf("dx=%f, dy=%f, exponent=%f\n", dx, dy, exponent);
    }

    particle.weight = std::exp(log_prob);
    weights_sum    += particle.weight;
  }
  // printf("weigth_sum: %f\n", weights_sum);

  // Normalize weights so that they represent probabilities
  weights.clear();
  for (Particle& particle : particles) {
    particle.weight /= weights_sum;
    weights.push_back(particle.weight);
  }
} // ParticleFilter::updateWeights

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  for (size_t i = 0; i < num_particles; i++) {
    int j = d(gen);
    new_particles.push_back(particles[j]);
  }
  particles = new_particles;
}

void ParticleFilter::setAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x      = sense_x;
  particle.sense_y      = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int>  v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream   ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream   ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
