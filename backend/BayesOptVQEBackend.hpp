#ifndef BAYESOPTVQEBACKEND_HPP_
#define BAYESOPTVQEBACKEND_HPP_

#include "VQEMinimizeTask.hpp"
#include "unsupported/Eigen/CXX11/Tensor"
#include "bayesopt/bayesopt.hpp"

namespace xacc {
namespace vqe {

class VQEBayesOptFunction: public bayesopt::ContinuousModel {
protected:
	int _dim = 1;
	std::shared_ptr<ComputeEnergyVQETask> computeTask;
public:

	VQEBayesOptFunction(bayesopt::Parameters par,std::shared_ptr<ComputeEnergyVQETask> task, const int dimension):
		ContinuousModel(dimension,par), computeTask(task), _dim(dimension) {}

	double evaluateSample(const boost::numeric::ublas::vector<double>& xin) {
	    	if (xin.size() != _dim) {
		      	xacc::error("WARNING: This only works for dimension "+std::to_string(_dim)+" inputs.");
	    	}

	       	// Map boost vector to eigen vector
	    	const double * data = &(xin.data()[0]);
	    	auto params = Eigen::Map<const Eigen::VectorXd>(data, _dim);
  
	      	return computeTask->execute(params).energy; //(x-0.3)*(x-0.3) + sin(20*x)*0.2;
	};

};

class BayesOptVQEBackend: public VQEBackend, public OptionsProvider {

public:

	virtual const VQETaskResult minimize(Eigen::VectorXd parameters);

	virtual const std::string name() const {
		return "vqe-bayesopt";
	}

	/**
	 * Return the description of this instance
	 * @return description The description of this object.
	 */
	virtual const std::string description() const {
		return "";
	}

	/**
	 * Return an empty options_description, this is for
	 * subclasses to implement.
	 */
	virtual std::shared_ptr<options_description> getOptions() {
		auto desc = std::make_shared<options_description>(
				"BayesOpt Options");
		desc->add_options()("bo-n-iter", value<std::string>(), "The number of bayesian optimization iterations.")
			("bo-noise", value<std::string>(), "The noise/signal ratio")
			("bo-learn_type", value<std::string>(), "The learning method for kernel hyperparameters")
			("bo-n-init-iter", value<std::string>(), "The number of initialization function calls.")
			("bo-verbose-level", value<std::string>(), "Verbose level for bayesian optimization")
			("bo-epsilon", value<std::string>(), "Epsilon-greedy strategy parameter");
		return desc;
	}

	virtual bool handleOptions(variables_map& map) {
		return false;
	}

};

}
}
#endif
