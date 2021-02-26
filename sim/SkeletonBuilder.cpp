#include "SkeletonBuilder.h"


namespace DPhy
{

double _default_damping_coefficient = JOINT_DAMPING;

Eigen::Vector3d Proj(const Eigen::Vector3d& u,const Eigen::Vector3d& v)
{
	Eigen::Vector3d proj;
	proj = u.dot(v)/u.dot(u)*u;
	return proj;	
}

Eigen::Isometry3d Orthonormalize(const Eigen::Isometry3d& T_old)
{
	Eigen::Isometry3d T;
	T.translation() = T_old.translation();
	Eigen::Vector3d v0,v1,v2;
	Eigen::Vector3d u0,u1,u2;
	v0 = T_old.linear().col(0);
	v1 = T_old.linear().col(1);
	v2 = T_old.linear().col(2);

	u0 = v0;
	u1 = v1 - Proj(u0,v1);
	u2 = v2 - Proj(u0,v2) - Proj(u1,v2);

	u0.normalize();
	u1.normalize();
	u2.normalize();

	T.linear().col(0) = u0;
	T.linear().col(1) = u1;
	T.linear().col(2) = u2;
	return T;
}
void 
SkeletonBuilder::
DeformBodyNode(const dart::dynamics::SkeletonPtr& skel,
	dart::dynamics::BodyNode* bn, 
	std::tuple<std::string, Eigen::Vector3d, double> deform) {
	
	auto shape_old = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
	
	auto inertia = bn->getInertia();
	inertia.setMass(inertia.getMass() * std::get<2>(deform));
	inertia.setMoment(shape_old->computeInertia(inertia.getMass()));
	bn->setInertia(inertia);

	if(std::get<1>(deform)(0) != 1 || std::get<1>(deform)(1) != 1 || std::get<1>(deform)(2) != 1) {
		auto box = dynamic_cast<dart::dynamics::BoxShape*>(shape_old);
		Eigen::Vector3d origin = box->getSize();
		Eigen::Vector3d size = origin.cwiseProduct(std::get<1>(deform));
		dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

		bn->removeAllShapeNodes();
	    bn->createShapeNodeWith<dart::dynamics::VisualAspect, dart::dynamics::CollisionAspect, dart::dynamics::DynamicsAspect>(shape);

		// auto props = bn->getParentJoint()->getJointProperties();
		// Eigen::Isometry3d joint_position = Eigen::Isometry3d::Identity(); 
		// Eigen::Isometry3d body_position =  Eigen::Isometry3d::Identity(); body_position.translation() = bn->getWorldTransform().translation();
		// body_position.translation()[1]= 0.5*size[1];

		// if(parent!=nullptr)
		// 	props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
		// props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;
		// props.mActuatorType = dart::dynamics::Joint::LOCKED;
		// bn->getParentJoint()->setProperties(props);
		
		auto props = bn->getParentJoint()->getJointProperties();
		Eigen::Vector3d translation = props.mT_ChildBodyToJoint.translation();

		for(int i = 0; i < 3; i++) {
			if(translation[i] != 0) {
				double sign = translation[i];
				sign = sign / fabs(sign);

				Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
				T.translation()[i] = sign * origin[i] * (std::get<1>(deform)[i] - 1) / 2.0;
				props.mT_ChildBodyToJoint = props.mT_ChildBodyToJoint * T;
				bn->getParentJoint()->setProperties(props);
			}
		}

		auto children = GetChildren(skel, bn);
		for(auto child : children) {
			props = child->getParentJoint()->getJointProperties();
			translation = props.mT_ParentBodyToJoint.translation();
			for(int i = 0; i < 3; i++) {
				if(translation[i] != 0) {

					double sign = translation[i];
					sign = sign / fabs(sign);

					Eigen::Isometry3d  T = Eigen::Isometry3d::Identity();
					T.translation()[i] = sign * origin[i] * (std::get<1>(deform)[i] - 1) / 2.0;

					props.mT_ParentBodyToJoint =  props.mT_ParentBodyToJoint * T;
					child->getParentJoint()->setProperties(props);
				}
			}
		}

	}
}

// torque limit map, position limit map
std::pair<dart::dynamics::SkeletonPtr, std::map<std::string, double>*>
SkeletonBuilder::
BuildFromFile(const std::string& filename){
	TiXmlDocument doc;
	if(!doc.LoadFile(filename)){
		std::cout << "Can't open file : " << filename << std::endl;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("Skeleton");
	
	std::string skelname = skeldoc->Attribute("name");
	dart::dynamics::SkeletonPtr skel = dart::dynamics::Skeleton::create(skelname);
	std::cout << skelname << std::endl;
	std::map<std::string, double>* torqueMap = new std::map<std::string, double>();
	std::map<std::string, Eigen::VectorXd>* positionMap = new std::map<std::string, Eigen::VectorXd>();

	for(TiXmlElement *body = skeldoc->FirstChildElement("Joint"); body != nullptr; body = body->NextSiblingElement("Joint")){
		// type
		std::string jointType = body->Attribute("type");
		// name
		std::string name = body->Attribute("name");
		// parent name
		std::string parentName = body->Attribute("parent_name");
		dart::dynamics::BodyNode *parent;
		if(!parentName.compare("None"))
			parent = nullptr;
		else
			parent = skel->getBodyNode(parentName);
		// size
		Eigen::Vector3d size = DPhy::string_to_vector3d(std::string(body->Attribute("size")));
		// body position
		TiXmlElement *bodyPosElem = body->FirstChildElement("BodyPosition");
		Eigen::Isometry3d bodyPosition;
		bodyPosition.setIdentity();
		if(bodyPosElem->Attribute("linear")!=nullptr)
			bodyPosition.linear() = DPhy::string_to_matrix3d(bodyPosElem->Attribute("linear"));
		bodyPosition.translation() = DPhy::string_to_vector3d(bodyPosElem->Attribute("translation"));
		bodyPosition = Orthonormalize(bodyPosition);
		// joint position
		TiXmlElement *jointPosElem = body->FirstChildElement("JointPosition");
		Eigen::Isometry3d jointPosition;
		jointPosition.setIdentity();
		if(jointPosElem->Attribute("linear")!=nullptr)
			jointPosition.linear() = DPhy::string_to_matrix3d(jointPosElem->Attribute("linear"));
		jointPosition.translation() = DPhy::string_to_vector3d(jointPosElem->Attribute("translation"));
		jointPosition = Orthonormalize(jointPosition);

		double torquelim = 1e6;
		TiXmlElement *torquelimElem = body->FirstChildElement("TorqueLimit");
		if(torquelimElem != nullptr) {
			torquelim = std::stod(torquelimElem->Attribute("norm"));
		}
		torqueMap->insert(std::pair<std::string, double>(name, torquelim));

		// shape : capsule, sphere, none, cylinder, box
		double shape_radius = 0;
		double shape_height = 0;
		int shape_type = 0;
		Eigen::Vector3d shape_direction, shape_offset, shape_size;
		shape_direction.setZero();
		shape_offset.setZero();
		shape_size.setZero();

		// capsule
		TiXmlElement *shapeElem = body->FirstChildElement("Capsule");
		if(shapeElem != nullptr){
			shape_direction = DPhy::string_to_vector3d(shapeElem->Attribute("direction"));
			shape_radius = atof(shapeElem->Attribute("radius"));
			shape_height = atof(shapeElem->Attribute("height"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = DPhy::string_to_vector3d(shapeElem->Attribute("offset"));;
			shape_type = 1;
		}

		// sphere
		shapeElem = body->FirstChildElement("Sphere");
		if(shapeElem != nullptr){
			shape_radius = atof(shapeElem->Attribute("radius"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = DPhy::string_to_vector3d(shapeElem->Attribute("offset"));;
			shape_type = 2;
		}

		// cylinder
		shapeElem = body->FirstChildElement("Cylinder");
		if(shapeElem != nullptr){
			shape_direction = DPhy::string_to_vector3d(shapeElem->Attribute("direction"));
			shape_radius = atof(shapeElem->Attribute("radius"));
			shape_height = atof(shapeElem->Attribute("height"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = DPhy::string_to_vector3d(shapeElem->Attribute("offset"));;
			shape_type = 3;
		}

		// box
		shapeElem = body->FirstChildElement("Box");
		if(shapeElem != nullptr){
			shape_size = DPhy::string_to_vector3d(shapeElem->Attribute("size"));
			if(shapeElem->Attribute("offset")!=nullptr)
				shape_offset = DPhy::string_to_vector3d(shapeElem->Attribute("offset"));;
			shape_type = 4;
		}

		// mass
		double mass = atof(body->Attribute("mass"));

		bool contact = true;
		
		if(!jointType.compare("FreeJoint") ){
			SkeletonBuilder::MakeFreeJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				mass,
				contact,
				shape_type,
				shape_radius, 
				shape_height, 
				shape_direction,
				shape_offset,
				shape_size
				);
		}

		else if(!jointType.compare("BallJoint")){
			// joint limit
			bool isLimitEnforced = false;
			Eigen::Vector3d upperLimit(1E6,1E6,1E6), lowerLimit(-1E6,-1E6,-1E6);
			if(jointPosElem->Attribute("upper")!=nullptr)
			{
				isLimitEnforced = true;
				upperLimit = DPhy::string_to_vector3d(jointPosElem->Attribute("upper"));
				lowerLimit = DPhy::string_to_vector3d(jointPosElem->Attribute("lower"));
			}

			SkeletonBuilder::MakeBallJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				contact,
				shape_type,
				shape_radius, 
				shape_height, 
				shape_direction,
				shape_offset,
				shape_size
				);
		}
		else if(!jointType.compare("RevoluteJoint")){
			// joint limit
			bool isLimitEnforced = false;
			double upperLimit(1E6), lowerLimit(-1E6);
			if(jointPosElem->Attribute("upper")!=nullptr)
			{
				isLimitEnforced = true;
				upperLimit = atof(jointPosElem->Attribute("upper"));
				lowerLimit = atof(jointPosElem->Attribute("lower"));
			}

			// axis
			Eigen::Vector3d axis = DPhy::string_to_vector3d(body->Attribute("axis"));

			SkeletonBuilder::MakeRevoluteJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				axis,
				contact
				);			
		}
		else if(!jointType.compare("PrismaticJoint")){
			// joint limit
			TiXmlElement *jointLimitElem = body->FirstChildElement("Limit");
			bool isLimitEnforced = false;
			double upperLimit, lowerLimit;
			if( jointLimitElem != nullptr ){
				isLimitEnforced = true;
				upperLimit = atof(jointLimitElem->Attribute("upper"));
				lowerLimit = atof(jointLimitElem->Attribute("lower"));
			}
			// axis
			Eigen::Vector3d axis = DPhy::string_to_vector3d(body->Attribute("axis"));

			SkeletonBuilder::MakePrismaticJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				isLimitEnforced,
				upperLimit,
				lowerLimit,
				mass,
				axis,
				contact
				);	

		}
		else if(!jointType.compare("WeldJoint")){
			SkeletonBuilder::MakeWeldJointBody(
				name,
				skel,
				parent,
				size,
				jointPosition,
				bodyPosition,
				mass,
				contact
				);			
		}

	}
	return std::pair<dart::dynamics::SkeletonPtr, std::map<std::string, double>*> (skel, torqueMap);
}

dart::
dynamics::
BodyNode* SkeletonBuilder::MakeFreeJointBall(
        const std::string& body_name,
        const dart::dynamics::SkeletonPtr& target_skel,
        dart::dynamics::BodyNode* const parent,
        const Eigen::Vector3d& size,
        const Eigen::Isometry3d& joint_position,
        const Eigen::Isometry3d& body_position,
        double mass,
        bool contact)
{
    double radius;
    dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::SphereShape>(new dart::dynamics::SphereShape(radius/*size*/));

    dart::dynamics::Inertia inertia;
    inertia.setMass(mass);
    inertia.setMoment(shape->computeInertia(mass));

    dart::dynamics::BodyNode* bn;
    dart::dynamics::FreeJoint::Properties props;
    props.mName = body_name;
    // props.mT_ChildBodyToJoint = joint_position;
    props.mT_ParentBodyToJoint = body_position;

    bn = target_skel->createJointAndBodyNodePair<dart::dynamics::FreeJoint>(
            parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;

    if(contact)
        bn->createShapeNodeWith<dart::dynamics::VisualAspect,dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
    else
        bn->createShapeNodeWith<dart::dynamics::VisualAspect, dart::dynamics::DynamicsAspect>(shape);
    bn->setInertia(inertia);
    return bn;
}

void 
SkeletonBuilder::
DeformSkeleton(const dart::dynamics::SkeletonPtr& skel, 
	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform) {
	for(auto d : deform) {
		for(int i=0;i<skel->getNumBodyNodes();i++)
		{
			auto bn = skel->getBodyNode(i);
			if(!bn->getName().compare(std::get<0>(d))) {
				DeformBodyNode(skel, bn, d);
				break;
			}
		}
	}
}

dart::
dynamics::
BodyNode* SkeletonBuilder::MakeFreeJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	double mass,
	bool contact,
	int shape_type,
	double shape_radius, 
	double shape_height, 
	Eigen::Vector3d shape_direction,
	Eigen::Vector3d shape_offset,
	Eigen::Vector3d shape_size
)
{
	dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

	double r = shape_radius;
	double h = shape_height;
	Eigen::Vector3d direction = shape_direction;
	dart::dynamics::ShapePtr shapeVisual;
	if(shape_type == 1)
		shapeVisual = std::shared_ptr<dart::dynamics::CapsuleShape>(new dart::dynamics::CapsuleShape(r, h));
	if(shape_type == 2)
		shapeVisual = std::shared_ptr<dart::dynamics::SphereShape>(new dart::dynamics::SphereShape(r));
	if(shape_type == 3)
		shapeVisual = std::shared_ptr<dart::dynamics::CylinderShape>(new dart::dynamics::CylinderShape(r, h));
	if(shape_type == 4)
		shapeVisual = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(shape_size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	dart::dynamics::BodyNode* bn;
	dart::dynamics::FreeJoint::Properties props;
	props.mName = body_name;
	// props.mT_ChildBodyToJoint = joint_position;
	props.mT_ParentBodyToJoint = body_position;

	bn = target_skel->createJointAndBodyNodePair<dart::dynamics::FreeJoint>(
		parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;

	if(contact){
		// bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
		bn->createShapeNodeWith<dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
	}
	else{
		bn->createShapeNodeWith<dart::dynamics::DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
	}
	bn->setInertia(inertia);
	return bn;
}

dart::
dynamics::
BodyNode* SkeletonBuilder::MakeBallJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	bool isLimitEnforced,
	const Eigen::Vector3d& upper_limit,
	const Eigen::Vector3d& lower_limit,
	double mass,
	bool contact,
	int shape_type,
	double shape_radius, 
	double shape_height, 
	Eigen::Vector3d shape_direction,
	Eigen::Vector3d shape_offset,
	Eigen::Vector3d shape_size
)
{
	dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

	double r = shape_radius;
	double h = shape_height;
	Eigen::Vector3d direction = shape_direction;
	dart::dynamics::ShapePtr shapeVisual;
	if(shape_type == 1)
		shapeVisual = std::shared_ptr<dart::dynamics::CapsuleShape>(new dart::dynamics::CapsuleShape(r, h));
	if(shape_type == 2)
		shapeVisual = std::shared_ptr<dart::dynamics::SphereShape>(new dart::dynamics::SphereShape(r));
	if(shape_type == 3)
		shapeVisual = std::shared_ptr<dart::dynamics::CylinderShape>(new dart::dynamics::CylinderShape(r, h));
	if(shape_type == 4)
		shapeVisual = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(shape_size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	dart::dynamics::BodyNode* bn;
	dart::dynamics::BallJoint::Properties props;
	props.mName = body_name;
	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<dart::dynamics::BallJoint>(
		parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;

	dart::dynamics::JointPtr jn = bn->getParentJoint();
	for(int i = 0; i < jn->getNumDofs(); i++){
		jn->getDof(i)->setDampingCoefficient(_default_damping_coefficient);
	}

	if(contact){
		// bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
		bn->createShapeNodeWith<dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
	}
	else{
		bn->createShapeNodeWith<dart::dynamics::DynamicsAspect>(shape);
		if(shape_type == 0){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
		}
		else if(shape_type == 1 || shape_type == 3){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.linear() = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction).toRotationMatrix();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 2){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		else if(shape_type == 4){
			bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shapeVisual);
			auto shapeNode = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
			Eigen::Isometry3d transform;
			transform.setIdentity();
			transform.translation() = shape_offset;
			shapeNode->setRelativeTransform(transform);
		}
		bn->createShapeNodeWith<dart::dynamics::VisualAspect>(shape);
	}
	bn->setInertia(inertia);

	if(isLimitEnforced){
		dart::dynamics::JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);

		for(int i = 0; i < 3; i++)
		{
			joint->setPositionUpperLimit(i, upper_limit[i]);
			joint->setPositionLowerLimit(i, lower_limit[i]);
		}
	}
	return bn;
}

dart::
dynamics::
BodyNode* SkeletonBuilder::MakeRevoluteJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	bool isLimitEnforced,
	double upper_limit,
	double lower_limit,
	double mass,
	const Eigen::Vector3d& axis,
	bool contact)
{
	dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	dart::dynamics::BodyNode* bn;
	dart::dynamics::RevoluteJoint::Properties props;
	props.mName = body_name;
	props.mAxis = axis;

	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<dart::dynamics::RevoluteJoint>(
		parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;

	dart::dynamics::JointPtr jn = bn->getParentJoint();
	for(int i = 0; i < jn->getNumDofs(); i++){
		jn->getDof(i)->setDampingCoefficient(_default_damping_coefficient);
	}

	if(contact)
		bn->createShapeNodeWith<dart::dynamics::VisualAspect,dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<dart::dynamics::VisualAspect, dart::dynamics::DynamicsAspect>(shape);
	bn->setInertia(inertia);

	if(isLimitEnforced){
		dart::dynamics::JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);
		joint->setPositionUpperLimit(0, upper_limit);
		joint->setPositionLowerLimit(0, lower_limit);
	}

	return bn;
}
dart::
dynamics::
BodyNode* SkeletonBuilder::MakePrismaticJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	bool isLimitEnforced,
	double upper_limit,
	double lower_limit,
	double mass,
	const Eigen::Vector3d& axis,
	bool contact)
{
	dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	dart::dynamics::BodyNode* bn;
	dart::dynamics::PrismaticJoint::Properties props;
	props.mName = body_name;
	props.mAxis = axis;

	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;
	props.mActuatorType = dart::dynamics::Joint::LOCKED;

	bn = target_skel->createJointAndBodyNodePair<dart::dynamics::PrismaticJoint>(
		parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;
	
	if(contact)
		bn->createShapeNodeWith<dart::dynamics::VisualAspect,dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<dart::dynamics::VisualAspect, dart::dynamics::DynamicsAspect>(shape);
	bn->setInertia(inertia);

	if(isLimitEnforced){
		dart::dynamics::JointPtr joint = bn->getParentJoint();
		joint->setPositionLimitEnforced(isLimitEnforced);
		joint->setPositionUpperLimit(0, upper_limit);
		joint->setPositionLowerLimit(0, lower_limit);
	}

	return bn;
}
dart::
dynamics::
BodyNode* SkeletonBuilder::MakeWeldJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& joint_position,
	const Eigen::Isometry3d& body_position,
	double mass,
	bool contact)
{
	dart::dynamics::ShapePtr shape = std::shared_ptr<dart::dynamics::BoxShape>(new dart::dynamics::BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	dart::dynamics::BodyNode* bn;
	dart::dynamics::WeldJoint::Properties props;
	props.mName = body_name;
	
	if(parent!=nullptr)
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*joint_position;
	props.mT_ChildBodyToJoint = body_position.inverse()*joint_position;

	bn = target_skel->createJointAndBodyNodePair<dart::dynamics::WeldJoint>(
		parent,props,dart::dynamics::BodyNode::AspectProperties(body_name)).second;
	
	if(contact)
		bn->createShapeNodeWith<dart::dynamics::VisualAspect,dart::dynamics::CollisionAspect,dart::dynamics::DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<dart::dynamics::VisualAspect, dart::dynamics::DynamicsAspect>(shape);
	bn->setInertia(inertia);

	return bn;
}



}