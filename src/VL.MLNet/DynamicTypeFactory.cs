using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reflection;
using System.Reflection.Emit;

namespace VL.MLNet
{
    // CREDITS : https://github.com/jonathancrozier/JC.Samples.DynamicProperties

    /// <summary>
    /// Generates new types with dynamically added properties
    /// </summary>
    class DynamicTypeFactory
    {
        // Fields
        private TypeBuilder _typeBuilder;
        
        // Readonlys
        private readonly AssemblyBuilder _assemblyBuilder;
        private readonly ModuleBuilder _moduleBuilder;

        // Constructor
        public DynamicTypeFactory()
        {
            var uniqueID     = Guid.NewGuid().ToString();
            var assemblyName = new AssemblyName(uniqueID);

            _assemblyBuilder = AssemblyBuilder.DefineDynamicAssembly(assemblyName, AssemblyBuilderAccess.RunAndCollect);
            _moduleBuilder   = _assemblyBuilder.DefineDynamicModule(uniqueID);
        }

        /// <summary>
        /// Creates a new Type based on the specified parent Type and attached dynamic properties
        /// </summary>
        /// <param name="parentType">The parent type to base the new type on</param>
        /// <param name="dynamicProperties">The collection of dynamic properties to attach to the new type</param>
        /// <returns>An extended Type with dynamic properties added to it</returns>
        public Type CreateNewTypeWithDynamicProperties(Type parentType, IEnumerable<DynamicProperty> dynamicProperties)
        {
            _typeBuilder = _moduleBuilder.DefineType(parentType.Name + Guid.NewGuid().ToString(), TypeAttributes.Public);
            _typeBuilder.SetParent(parentType);

            foreach (DynamicProperty property in dynamicProperties)
                AddDynamicPropertyToType(property);

            return _typeBuilder.CreateType();
        }

        private void AddDynamicPropertyToType(DynamicProperty dynamicProperty)
        {
            Type propertyType   = dynamicProperty.SystemType;
            // string propertyName = $"{nameof(DynamicProperty)}_{dynamicProperty.PropertyName}";
            string propertyName = $"{dynamicProperty.PropertyName}";
            string fieldName    = $"_{propertyName}";

            FieldBuilder fieldBuilder = _typeBuilder.DefineField(fieldName, propertyType, FieldAttributes.Private);

            // Getters and setters require a special set of attributes
            MethodAttributes getSetAttributes = MethodAttributes.Public | MethodAttributes.SpecialName | MethodAttributes.HideBySig;

            // Define the GET accessor method
            MethodBuilder getMethodBuilder   = _typeBuilder.DefineMethod($"get_{propertyName}", getSetAttributes, propertyType, Type.EmptyTypes);
            ILGenerator propertyGetGenerator = getMethodBuilder.GetILGenerator();
            propertyGetGenerator.Emit(OpCodes.Ldarg_0);
            propertyGetGenerator.Emit(OpCodes.Ldfld, fieldBuilder);
            propertyGetGenerator.Emit(OpCodes.Ret);

            // Define the SET accessor method
            MethodBuilder setMethodBuilder   = _typeBuilder.DefineMethod($"set_{propertyName}", getSetAttributes, null, new Type[] { propertyType });
            ILGenerator propertySetGenerator = setMethodBuilder.GetILGenerator();
            propertySetGenerator.Emit(OpCodes.Ldarg_0);
            propertySetGenerator.Emit(OpCodes.Ldarg_1);
            propertySetGenerator.Emit(OpCodes.Stfld, fieldBuilder);
            propertySetGenerator.Emit(OpCodes.Ret);

            // Wraps the two methods we created above to a PropertyBuilder and their get/set behaviors
            PropertyBuilder propertyBuilder = _typeBuilder.DefineProperty(propertyName, PropertyAttributes.HasDefault, propertyType, null);
            propertyBuilder.SetGetMethod(getMethodBuilder);
            propertyBuilder.SetSetMethod(setMethodBuilder);
        }
    }

    public class DynamicProperty
    {
        /// <summary>
        /// The name of the property
        /// </summary>
        public string PropertyName { get; set; }

        /// <summary>
        /// The display-name of the property for the end user
        /// </summary>
        public string DisplayName { get; set; }

        /// <summary>
        /// Name of the underlying System Type of property
        /// </summary>
        public string SystemTypeName { get; set; }

        /// <summary>
        /// Underlyin System Type of the property
        /// </summary>
        public Type SystemType => Type.GetType(SystemTypeName);
    }
}
